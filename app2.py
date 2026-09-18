from datetime import date, datetime, timedelta
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
import requests
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN Y BASE DE DATOS
# ---------------------------------------------------------
DB_FILE = "pvpc_prices.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pvpc (
            date TEXT,
            hour INT,
            datetime_utc TEXT,
            price_eur_kwh REAL,
            fetch_timestamp TEXT,
            source TEXT,
            PRIMARY KEY (date, hour)
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------
# EXTRACCIÓN DE DATOS (ESIOS ARCHIVO 70 - SIN TOKEN)
# ---------------------------------------------------------
def fetch_pvpc_ree_public(start_date: date, end_date: date):
    """Descarga el PVPC desde el Archivo 70 público de ESIOS y reporta estados."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    current_date = start_date
    records = []
    now_str = datetime.now().isoformat()
    today_date = date.today()
    messages = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api.esios.ree.es/archives/70/download_json?date={date_str}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "PVPC" in data:
                    for item in data["PVPC"]:
                        hour = int(item["Hora"].split("-")[0])
                        price_mwh_str = item["PCB"].replace(",", ".")
                        price_kwh = float(price_mwh_str) / 1000.0
                        dt_iso = f"{date_str}T{hour:02d}:00:00"

                        records.append(
                            (
                                date_str,
                                hour,
                                dt_iso,
                                price_kwh,
                                now_str,
                                "ESIOS_ARCHIVE_70",
                            )
                        )
                    messages.append(
                        ("success", f"✅ Precios del {date_str} cargados.")
                    )
            elif (
                response.status_code in (404, 500) and current_date > today_date
            ):
                messages.append(
                    (
                        "info",
                        (
                            f"ℹ️ {date_str}: Precios de mañana no disponibles"
                            " aún (disponibles ~20:30 CET)."
                        ),
                    )
                )
            else:
                messages.append(
                    (
                        "error",
                        (
                            f"⚠️ Error {response.status_code} al consultar el"
                            f" día {date_str}"
                        ),
                    )
                )

        except requests.exceptions.Timeout:
            messages.append(
                ("error", f"⏳ Timeout al consultar el día {date_str}")
            )
        except Exception as e:
            messages.append(
                ("error", f"❌ Error al procesar el día {date_str}: {e}")
            )

        current_date += timedelta(days=1)

    if records:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR IGNORE INTO pvpc (date, hour, datetime_utc, price_eur_kwh, fetch_timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            records,
        )
        conn.commit()
        conn.close()

    return len(records), messages


def load_data_from_db(start_date: date, end_date: date):
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT date, hour, datetime_utc, price_eur_kwh
        FROM pvpc
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC, hour ASC
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime_utc"])
    return df


# ---------------------------------------------------------
# ALGORITMO DE OPTIMIZACIÓN MULTI-ELECTRODOMÉSTICO (MILP)
# ---------------------------------------------------------
def solve_multi_appliance_schedule(
    df_prices: pd.DataFrame, appliances_config: dict, max_power_kw: float
):
    N = len(df_prices)
    if N == 0:
        return None, "Sin datos para optimizar"

    prices = df_prices["price_eur_kwh"].tolist()
    datetimes = df_prices["datetime"].tolist()

    prob = pulp.LpProblem("Optimizador_PVPC_Multi", pulp.LpMinimize)

    x = {}
    s = {}

    active_appliances = [
        name for name, cfg in appliances_config.items() if cfg["active"]
    ]

    for name in active_appliances:
        cfg = appliances_config[name]
        duration = cfg["hours"]
        if duration > N:
            return (
                None,
                (
                    f"El aparato '{name}' requiere {duration}h, pero el rango"
                    f" solo tiene {N}h seleccionadas."
                ),
            )

        for t in range(N):
            x[name, t] = pulp.LpVariable(
                f"state_{name}_{t}", cat=pulp.LpBinary
            )

        if cfg["continuous"]:
            for t in range(N - duration + 1):
                s[name, t] = pulp.LpVariable(
                    f"start_{name}_{t}", cat=pulp.LpBinary
                )

            prob += (
                pulp.lpSum(s[name, t] for t in range(N - duration + 1)) == 1
            )

            for t in range(N):
                valid_starts = [
                    s[name, k]
                    for k in range(
                        max(0, t - duration + 1), min(t + 1, N - duration + 1)
                    )
                ]
                prob += x[name, t] == pulp.lpSum(valid_starts)
        else:
            prob += pulp.lpSum(x[name, t] for t in range(N)) == duration

    for t in range(N):
        prob += (
            pulp.lpSum(
                x[name, t] * appliances_config[name]["power"]
                for name in active_appliances
            )
            <= max_power_kw
        )

    prob += pulp.lpSum(
        x[name, t] * appliances_config[name]["power"] * prices[t]
        for name in active_appliances
        for t in range(N)
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return (
            None,
            (
                "No se encontró solución factible. Prueba a aumentar la"
                " potencia máxima o reducir horas necesarias."
            ),
        )

    schedule_df = df_prices[["datetime", "price_eur_kwh"]].copy()
    schedule_df["potencia_total_kw"] = 0.0

    summary_list = []

    for name in active_appliances:
        kw = appliances_config[name]["power"]
        schedule_df[name] = [
            pulp.value(x[name, t]) * kw for t in range(N)
        ]
        schedule_df["potencia_total_kw"] += schedule_df[name]

        hours_active = [t for t in range(N) if pulp.value(x[name, t]) > 0.5]
        first_start = datetimes[hours_active[0]]
        last_end = datetimes[hours_active[-1]] + timedelta(hours=1)
        cost = sum(
            pulp.value(x[name, t]) * kw * prices[t] for t in range(N)
        )

        summary_list.append(
            {
                "Aparato": name,
                "Potencia (kW)": kw,
                "Horas Requeridas": appliances_config[name]["hours"],
                "Modo": (
                    "Seguidas"
                    if appliances_config[name]["continuous"]
                    else "Flexibles"
                ),
                "Inicio Estimado": first_start.strftime("%d/%m %H:00"),
                "Fin Estimado": last_end.strftime("%d/%m %H:00"),
                "Coste (€)": f"{cost:.3f} €",
            }
        )

    total_cost = sum(
        pulp.value(x[name, t])
        * appliances_config[name]["power"]
        * prices[t]
        for name in active_appliances
        for t in range(N)
    )

    return (schedule_df, pd.DataFrame(summary_list), total_cost), "OK"


# ---------------------------------------------------------
# INTERFAZ STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="PVPC Multi-Appliance Optimizer", layout="wide", page_icon="⚡"
)
st.title("⚡ Optimizador de Cargas Domésticas PVPC")

# --- BARRA LATERAL ---
st.sidebar.header("📥 Descarga de Precios")

if st.sidebar.button("🔄 Descargar Hoy + Mañana", use_container_width=True):
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    with st.spinner("Consultando ESIOS..."):
        _, msgs = fetch_pvpc_ree_public(today, tomorrow)
        for msg_type, text in msgs:
            if msg_type == "success":
                st.sidebar.success(text)
            elif msg_type == "info":
                st.sidebar.info(text)
            else:
                st.sidebar.error(text)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Potencia Contratada")
max_power_input = st.sidebar.number_input(
    "Límite Máximo (kW):", min_value=1.0, max_value=15.0, value=4.4, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.header("🧺 Electrodomésticos")

appliances = {}

# Coche Eléctrico
with st.sidebar.expander("🚘 Coche Eléctrico", expanded=True):
    use_car = st.checkbox("Activar Coche", value=True)
    car_kw = st.number_input(
        "Potencia (kW)",
        min_value=0.5,
        max_value=11.0,
        value=2.2,
        step=0.1,
        key="car_kw",
    )
    car_h = st.number_input(
        "Horas necesarias", min_value=1, max_value=24, value=5, key="car_h"
    )
    car_cont = st.checkbox("Horas seguidas", value=True, key="car_cont")
    appliances["Coche Eléctrico"] = {
        "active": use_car,
        "power": car_kw,
        "hours": car_h,
        "continuous": car_cont,
    }

# Lavadora
with st.sidebar.expander("🧺 Lavadora", expanded=False):
    use_wash = st.checkbox("Activar Lavadora", value=True)
    wash_kw = st.number_input(
        "Potencia (kW)",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.1,
        key="wash_kw",
    )
    wash_h = st.number_input(
        "Horas necesarias", min_value=1, max_value=12, value=2, key="wash_h"
    )
    wash_cont = st.checkbox("Horas seguidas", value=True, key="wash_cont")
    appliances["Lavadora"] = {
        "active": use_wash,
        "power": wash_kw,
        "hours": wash_h,
        "continuous": wash_cont,
    }

# Secadora
with st.sidebar.expander("🌀 Secadora", expanded=False):
    use_dryer = st.checkbox("Activar Secadora", value=False)
    dryer_kw = st.number_input(
        "Potencia (kW)",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1,
        key="dryer_kw",
    )
    dryer_h = st.number_input(
        "Horas necesarias", min_value=1, max_value=12, value=2, key="dryer_h"
    )
    dryer_cont = st.checkbox("Horas seguidas", value=True, key="dryer_cont")
    appliances["Secadora"] = {
        "active": use_dryer,
        "power": dryer_kw,
        "hours": dryer_h,
        "continuous": dryer_cont,
    }

# Lavavajillas
with st.sidebar.expander("🍽️ Lavavajillas", expanded=False):
    use_dish = st.checkbox("Activar Lavavajillas", value=True)
    dish_kw = st.number_input(
        "Potencia (kW)",
        min_value=0.5,
        max_value=4.0,
        value=1.2,
        step=0.1,
        key="dish_kw",
    )
    dish_h = st.number_input(
        "Horas necesarias", min_value=1, max_value=12, value=2, key="dish_h"
    )
    dish_cont = st.checkbox("Horas seguidas", value=True, key="dish_cont")
    appliances["Lavavajillas"] = {
        "active": use_dish,
        "power": dish_kw,
        "hours": dish_h,
        "continuous": dish_cont,
    }

# Horno
with st.sidebar.expander("🍳 Horno", expanded=False):
    use_oven = st.checkbox("Activar Horno", value=False)
    oven_kw = st.number_input(
        "Potencia (kW)",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1,
        key="oven_kw",
    )
    oven_h = st.number_input(
        "Horas necesarias", min_value=1, max_value=12, value=2, key="oven_h"
    )
    oven_cont = st.checkbox("Horas seguidas", value=True, key="oven_cont")
    appliances["Horno"] = {
        "active": use_oven,
        "power": oven_kw,
        "hours": oven_h,
        "continuous": oven_cont,
    }

# --- PANEL PRINCIPAL ---
today = date.today()
tomorrow = today + timedelta(days=1)

# period_option = st.radio(
#     "Ver planificación para:",
#     ["Hoy", "Mañana", "Hoy + Mañana", "Personalizado"],
#     horizontal=True,
# )

# Comprobación automática de la disponibilidad del día de mañana:
df_tomorrow = load_data_from_db(tomorrow, tomorrow)

if not df_tomorrow.empty:
    view_start, view_end = tomorrow, tomorrow
    df = df_tomorrow
else:
    view_start, view_end = today, today
    df = load_data_from_db(today, today)
    if not df.empty:
        st.info(
            "ℹ️ Los precios de mañana aún no están disponibles (disponibles"
            " ~20:30 CET)."
        )

if df.empty:
    st.info(
        f"No hay precios guardados en la base de datos para hoy ({today}). Usa"
        " el botón **🔄 Descargar Hoy + Mañana** de la barra lateral."
    )
else:
    # Métricas (precios redondeados a 2 decimales y colores/flechas personalizados)
    col1, col2, col3, col4 = st.columns(4)
    min_row = df.loc[df["price_eur_kwh"].idxmin()]
    max_row = df.loc[df["price_eur_kwh"].idxmax()]
    avg_price = df["price_eur_kwh"].mean()

    # Mínimo: Flecha hacia abajo (▼) en verde
    col1.metric(
        "Precio Mínimo",
        f"{min_row['price_eur_kwh']:.2f} €/kWh",
        f"- Hora {min_row['hour']}:00 ({min_row['date']})",
        delta_color="inverse",
    )
    # Máximo: Flecha hacia arriba (▲) en rojo
    col2.metric(
        "Precio Máximo",
        f"{max_row['price_eur_kwh']:.2f} €/kWh",
        f"Hora {max_row['hour']}:00 ({max_row['date']})",
        delta_color="inverse",
    )
    col3.metric("Precio Medio", f"{avg_price:.2f} €/kWh")
    col4.metric("Horas en Rango", f"{len(df)} h")

    st.markdown("---")
    st.subheader("🎯 Planificación Distribuidora Óptima")

    result, status_msg = solve_multi_appliance_schedule(
        df, appliances, max_power_input
    )

    if result is None:
        st.error(f"⚠️ {status_msg}")
    else:
        schedule_df, summary_df, total_cost = result

        st.success(f"💰 **Coste estimado del plan:** `{total_cost:.3f} €`")
        st.dataframe(summary_df, use_container_width=True)

        # --- GRÁFICO COMBINADO ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        color_map = {
            "Coche Eléctrico": "#1f77b4",
            "Lavadora": "#ff7f0e",
            "Secadora": "#9467bd",
            "Lavavajillas": "#2ca02c",
            "Horno": "#d62728",
        }

        for app_name in appliances:
            if appliances[app_name]["active"] and app_name in schedule_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=schedule_df["datetime"],
                        y=schedule_df[app_name],
                        name=f"{app_name} (kW)",
                        marker_color=color_map.get(app_name, "#8c564b"),
                    ),
                    secondary_y=False,
                )

        fig.add_trace(
            go.Scatter(
                x=schedule_df["datetime"],
                y=[max_power_input] * len(schedule_df),
                mode="lines",
                name=f"Límite Potencia ({max_power_input} kW)",
                line=dict(color="red", width=2, dash="dash"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=schedule_df["datetime"],
                y=schedule_df["price_eur_kwh"],
                mode="lines+markers",
                name="Precio PVPC (€/kWh)",
                line=dict(color="#2b5c8f", width=2),
                marker=dict(size=4),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Distribución de Cargas vs Potencia Contratada y PVPC",
            barmode="stack",
            xaxis_title="Fecha y Hora",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(
            title_text="<b>Potencia Consumida (kW)</b>", secondary_y=False
        )
        fig.update_yaxes(
            title_text="<b>Precio PVPC (€/kWh)</b>", secondary_y=True
        )

        st.plotly_chart(fig, use_container_width=True)