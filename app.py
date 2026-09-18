from datetime import date, datetime, timedelta
import sqlite3
import pandas as pd
import plotly.graph_objects as go
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
# EXTRACCIÓN SIN TOKEN (CON USER-AGENT FALSO)
# ---------------------------------------------------------


def fetch_pvpc_ree_public(start_date: date, end_date: date):
    """Descarga el PVPC desde el Archivo 70 de ESIOS (fichero JSON público oficial)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
            " Safari/537.36"
        )
    }

    current_date = start_date
    records = []
    now_str = datetime.now().isoformat()
    today_date = date.today()

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api.esios.ree.es/archives/70/download_json?date={date_str}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "PVPC" in data:
                    for item in data["PVPC"]:
                        # "Hora" viene en formato "00-01", "01-02"...
                        hour = int(item["Hora"].split("-")[0])

                        # "PCB" (Península/Canarias/Baleares) viene como string "120,45" en €/MWh
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
            elif response.status_code in (404, 500) and current_date > today_date:
                st.warning(
                    f"⚠️ {date_str}: Precios no publicados aún (disponibles"
                    " ~20:30 CET)."
                )
            else:
                st.error(
                    f"⚠️ Error {response.status_code} al consultar el día"
                    f" {date_str}"
                )

        except requests.exceptions.Timeout:
            st.error(f"⏳ Timeout al consultar {date_str}")
        except Exception as e:
            st.error(f"❌ Error al procesar {date_str}: {e}")

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

    return len(records)

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
# INTERFAZ STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="PVPC Explorer & Optimizer", layout="wide", page_icon="⚡"
)
st.title("⚡ Explorador y Optimizador PVPC")

# --- BARRA LATERAL: Descargas ---
st.sidebar.header("📥 Descarga de Datos (Sin Token)")

if st.sidebar.button("Obtener Precios Hoy / Mañana"):
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    with st.spinner("Consultando API pública de REE..."):
        count = fetch_pvpc_ree_public(today, tomorrow)
        st.sidebar.success(f"Registros guardados: {count}")

st.sidebar.markdown("---")
st.sidebar.subheader("Descarga Histórica")
hist_start = st.sidebar.date_input(
    "Fecha Inicio", datetime.now().date() - timedelta(days=7)
)
hist_end = st.sidebar.date_input("Fecha Fin", datetime.now().date())

if st.sidebar.button("Descargar Histórico"):
    if hist_start <= hist_end:
        with st.spinner("Descargando histórico..."):
            count = fetch_pvpc_ree_public(hist_start, hist_end)
            st.sidebar.success(f"Procesados {count} registros en BD.")
    else:
        st.sidebar.error(
            "La fecha de inicio debe ser menor o igual a la de fin."
        )

# --- PANEL PRINCIPAL ---
st.subheader("🗓️ Selección de Rango de Análisis")
col_date1, col_date2 = st.columns(2)
with col_date1:
    view_start = st.date_input("Desde", datetime.now().date())
with col_date2:
    view_end = st.date_input("Hasta", datetime.now().date())

df = load_data_from_db(view_start, view_end)

if df.empty:
    st.info(
        "No hay datos guardados para el rango seleccionado. Usa la barra lateral para descargarlos."
    )
else:
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    min_row = df.loc[df["price_eur_kwh"].idxmin()]
    max_row = df.loc[df["price_eur_kwh"].idxmax()]
    avg_price = df["price_eur_kwh"].mean()

    col1.metric(
        "Precio Mínimo",
        f"{min_row['price_eur_kwh']:.5f} €/kWh",
        f"Hora {min_row['hour']}:00 ({min_row['date']})",
    )
    col2.metric(
        "Precio Máximo",
        f"{max_row['price_eur_kwh']:.5f} €/kWh",
        f"Hora {max_row['hour']}:00 ({max_row['date']})",
    )
    col3.metric("Precio Medio", f"{avg_price:.5f} €/kWh")
    col4.metric("Total Horas en BD", f"{len(df)} h")

    st.markdown("---")

    # Optimizador de ventana
    st.subheader("🎯 Optimizador de Ventana de Consumo Continuo")

    window_hours = st.slider(
        "Horas consecutivas requeridas (Ej: 5h Coche, 2h Lavadora):",
        min_value=1,
        max_value=12,
        value=3,
    )

    if len(df) >= window_hours:
        df["rolling_avg"] = (
            df["price_eur_kwh"].rolling(window=window_hours).mean()
        )

        best_end_idx = df["rolling_avg"].idxmin()
        best_start_idx = best_end_idx - window_hours + 1

        best_window = df.loc[best_start_idx:best_end_idx]
        best_start_time = best_window.iloc[0]["datetime"]
        best_end_time = best_window.iloc[-1]["datetime"] + timedelta(hours=1)
        best_avg_price = best_window["price_eur_kwh"].mean()

        st.success(
            f"💡 **Mejor ventana de {window_hours}h:** "
            f"Desde **{best_start_time.strftime('%Y-%m-%d %H:00')}** "
            f"hasta **{best_end_time.strftime('%Y-%m-%d %H:00')}** "
            f"| **Precio medio:** `{best_avg_price:.5f} €/kWh`"
        )

        # Gráfica
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["price_eur_kwh"],
                mode="lines+markers",
                name="Precio PVPC (€/kWh)",
                line=dict(color="#2b5c8f", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=best_window["datetime"],
                y=best_window["price_eur_kwh"],
                mode="lines+markers",
                name=f"Mejor Ventana ({window_hours}h)",
                line=dict(color="#00cc96", width=4),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Evolución del Precio PVPC y Ventana Óptima Destacada",
            xaxis_title="Fecha y Hora",
            yaxis_title="€ / kWh",
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"Se necesitan al menos {window_hours} horas de datos guardados para calcular la ventana."
        )