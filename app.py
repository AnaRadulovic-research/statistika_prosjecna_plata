import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import lognorm

# --- KONFIGURACIJA STRANICE ---
st.set_page_config(
    page_title="Analiza prihoda Crne Gore",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PRILAGOĐENI DIZAJN (CSS) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #B22222; /* Crnogorska crvena */
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #D4AF37; /* Zlatna */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #b8dbe9;
        margin-top: 10px;
    }
    .gini-box {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ffcccc;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- POMOĆNE FUNKCIJE ---

def calculate_gini(array):
    """Izračunava Gini koeficijent za dati niz podataka."""
    array = np.array(array).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

@st.cache_data
def generate_salary_data(n_samples=5000, mean_salary=1000, gini_target=0.32):
    """
    Generiše log-normalnu distribuciju plata kako bi se simulirala nejednakost.
    """
    sigma = gini_target * 2.0 
    mu = np.log(mean_salary) - (sigma**2 / 2)
    data = lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_samples)
    
    sectors = [
        ('IT i Finansije', 1.6),     
        ('Državna uprava', 1.1),     
        ('Građevinarstvo', 0.9),     
        ('Trgovina i Usluge', 0.7),  
        ('Poljoprivreda', 0.6)       
    ]
    
    df_list = []
    for sector, multiplier in sectors:
        sector_data = data[:int(n_samples/len(sectors))] * multiplier
        # Dodavanje šuma i minimalne zarade
        sector_data = sector_data + np.random.normal(0, 50, len(sector_data))
        sector_data = np.maximum(sector_data, 450) # Minimalna plata u CG
        
        temp_df = pd.DataFrame({
            'Neto plata (€)': np.round(sector_data, 0),
            'Sektor': sector
        })
        df_list.append(temp_df)
    
    return pd.concat(df_list).reset_index(drop=True)

def lorenz_curve(data):
    sorted_data = np.sort(data)
    lorenz_y = np.cumsum(sorted_data) / np.sum(sorted_data)
    lorenz_y = np.insert(lorenz_y, 0, 0)
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    return lorenz_x, lorenz_y

# --- GLAVNA LOGIKA APLIKACIJE ---

def main():
    st.markdown('<div class="main-header">Realnost plata u Crnoj Gori 🇲🇪</div>', unsafe_allow_html=True)
    st.markdown("### Zašto vas 'prosječna plata' često vara?")
    
    st.markdown("""
    Većina izvještaja se fokusira na **Prosjek (Aritmetičku sredinu)**. Ali, da li to odražava vašu stvarnost? 
    Ovaj alat simulira ekonomiju kako bi objasnio razliku između statističkih iluzija i stvarnog stanja u novčanicima.
    """)
    
    # Bočna traka (Sidebar)
    st.sidebar.header("⚙️ Podešavanja simulacije")
    input_mean = st.sidebar.slider("Zvanična prosječna neto plata (€)", 800, 1500, 1004, step=10)
    
    gini_selection = st.sidebar.select_slider(
        "Nivo nejednakosti (Gini indeks)", 
        options=[0.25, 0.32, 0.45],
        format_func=lambda x: f"{x} (Skandinavija)" if x==0.25 else (f"{x} (CG Procjena)" if x==0.32 else f"{x} (Visoka nejednakost)")
    )
    
    # Generisanje podataka
    df = generate_salary_data(mean_salary=input_mean, gini_target=gini_selection)
    salaries = df['Neto plata (€)'].values
    
    # Proračuni
    calc_mean = np.mean(salaries)
    calc_median = np.median(salaries)
    real_gini = calculate_gini(salaries)
    
    # --- RED SA METRIKAMA ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card"><h3>Prosjek (Mean) 📈</h3><h2>€{calc_mean:,.0f}</h2><p>Napumpan visokim zaradama</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h3>Sredina (Median) 📉</h3><h2>€{calc_median:,.0f}</h2><p>50% građana zarađuje manje od ovoga</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><h3>Gini Indeks ⚖️</h3><h2>{real_gini:.2f}</h2><p>0 = Jednakost, 1 = Maksimalna nepravda</p></div>""", unsafe_allow_html=True)
    
    st.write("---")
    
    # --- TABOVI ZA VIZUELIZACIJU ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribucija", "⚖️ Gini i Nejednakost", "💰 Raspodjela novca", "🏭 Sektori"])
    
    # 1. TAB: DISTRIBUCIJA
    with tab1:
        st.subheader("Histogram distribucije prihoda")
        fig_dist = px.histogram(df, x="Neto plata (€)", nbins=60, color_discrete_sequence=['#636EFA'], opacity=0.7,
                               labels={'Neto plata (€)': 'Neto plata u eurima'})
        fig_dist.add_vline(x=calc_mean, line_dash="dash", line_color="red", annotation_text="Prosjek")
        fig_dist.add_vline(x=calc_median, line_dash="solid", line_color="green", annotation_text="Medijana")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown(f"""
        <div class="explanation-box">
        <b>Objašnjenje za običnog građanina:</b><br>
        Zamislite sobu sa 10 ljudi. Ako uđe milijarder, "prosječno" bogatstvo sobe skače na milione, ali ostalih 10 ljudi su i dalje siromašni.
        <br><b>Zelena linija (Medijana)</b> predstavlja osobu koja je tačno u sredini niza. <b>Crvena linija (Prosjek)</b> je povučena udesno zbog malog broja veoma visokih plata.
        </div>
        """, unsafe_allow_html=True)

    # 2. TAB: GINI I LORENZ
    with tab2:
        st.subheader("Vizuelizacija Gini koeficijenta")
        
        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            lorenz_x, lorenz_y = lorenz_curve(salaries)
            fig_lorenz = go.Figure()
            fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Savršena jednakost', line=dict(dash='dash', color='gray')))
            fig_lorenz.add_trace(go.Scatter(x=lorenz_x, y=lorenz_y, mode='lines', name='Stvarnost', line=dict(color='red', width=3)))
            
            fig_lorenz.add_trace(go.Scatter(x=lorenz_x, y=lorenz_y, fill='tozeroy', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), showlegend=False))
            
            fig_lorenz.update_layout(xaxis_title="% Populacije", yaxis_title="% Ukupnog novca", height=400)
            st.plotly_chart(fig_lorenz, use_container_width=True)

        with col_g2:
            st.markdown(f"""
            <div class="gini-box">
            <h3>Gini Indeks: {real_gini:.2f}</h3>
            <p><b>Šta ovo znači?</b></p>
            <ul>
            <li><b>0.0:</b> Savršena jednakost (svi imaju isto)</li>
            <li><b>0.30:</b> Zdrava ekonomija (Prosjek EU)</li>
            <li><b>0.50+:</b> Opasna nejednakost</li>
            </ul>
            <br>
            Što je "stomak" crvene krive veći, to je nejednakost u društvu izraženija.
            </div>
            """, unsafe_allow_html=True)

    # 3. TAB: DECILI (KO UZIMA NOVAC?)
    with tab3:
        st.subheader("Udio u ukupnom novcu po grupama (Decili)")
        
        # ISPRAVKA: Uklonjen je problem sa fiksnim labelima
        try:
            # Pokušaj da napraviš 10 decila
            df['Decil'] = pd.qcut(df['Neto plata (€)'], 10, duplicates='drop')
        except Exception as e:
            st.error(f"Greška pri kreiranju decila: {e}")
            # Fallback - napravi manje grupa ako je potrebno
            df['Decil'] = pd.qcut(df['Neto plata (€)'], 5, duplicates='drop')
        
        # Grupišemo po decilima
        decile_sum = df.groupby('Decil', observed=True)['Neto plata (€)'].sum().reset_index()
        decile_sum['Udio (%)'] = (decile_sum['Neto plata (€)'] / decile_sum['Neto plata (€)'].sum()) * 100
        
        # Kreiraj prilagođene labele za X-osu
        decile_sum['Grupa'] = [f"Grupa {i+1}" for i in range(len(decile_sum))]
        
        fig_bar = px.bar(
            decile_sum, 
            x='Grupa', 
            y='Udio (%)', 
            title="Koliki dio 'ukupnog kolača' plata dobija svaka grupa stanovništva?",
            color='Udio (%)',
            color_continuous_scale='Reds',
            labels={'Udio (%)': 'Udio u ukupnom fondu (%)', 'Grupa': 'Grupe stanovništva (po prihodima)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        if len(decile_sum) >= 2:
            top_share = decile_sum.iloc[-1]['Udio (%)']
            bottom_share = decile_sum.iloc[0]['Udio (%)']
            ratio = top_share / bottom_share if bottom_share > 0 else 0
            
            st.markdown(f"""
            <div class="explanation-box">
            <b>Provjera realnosti:</b><br>
            Najbogatija grupa u ovoj simulaciji uzima <b>{top_share:.1f}%</b> ukupnog novca od plata.<br>
            Najsiromašnija grupa uzima svega <b>{bottom_share:.1f}%</b>.<br><br>
            To znači da najbogatija grupa zarađuje <b>{ratio:.1f} puta više</b> nego najsiromašnija grupa zajedno.
            </div>
            """, unsafe_allow_html=True)

    # 4. TAB: SEKTORI
    with tab4:
        st.subheader("Poređenje po sektorima")
        fig_box = px.box(df, x="Sektor", y="Neto plata (€)", color="Sektor",
                        labels={'Neto plata (€)': 'Plata (€)'})
        st.plotly_chart(fig_box, use_container_width=True)
        st.write("Box-plot grafik pokazuje raspon plata unutar svakog sektora, uključujući i ekstremno visoke plate (tačkice iznad kutija).")

    # --- FOOTER ---
    st.write("---")
    st.markdown("Razvijeno u okviru Data Science portfolija koristeći Python i Streamlit.")

if __name__ == "__main__":
    main()
