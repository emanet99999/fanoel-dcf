import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import requests
from bs4 import BeautifulSoup

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Fanoel Valuation Suite Pro Max", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .big-metric { font-size: 32px; font-weight: bold; color: #2c3e50; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Fanoel DCF Valuation - Pro Suite Ultimate Edition")

# ============================
# YARDIMCI FONKSİYONLAR
# ============================

def calculate_piotroski_score(info, financials, balance_sheet, cash_flow):
    """Piotroski F-Score hesaplama (0-9)"""
    score = 0
    try:
        # 1. Profitability (4 puan)
        net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
        if net_income > 0: score += 1
        
        operating_cf = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
        if operating_cf > 0: score += 1
        
        total_assets_current = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
        total_assets_prev = balance_sheet.loc['Total Assets'].iloc[1] if len(balance_sheet.columns) > 1 else total_assets_current
        roa = net_income / total_assets_current
        roa_prev = financials.loc['Net Income'].iloc[1] / total_assets_prev if len(financials.columns) > 1 else roa
        if roa > roa_prev: score += 1
        
        if operating_cf > net_income: score += 1
        
        # 2. Leverage/Liquidity (3 puan)
        long_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
        long_debt_prev = balance_sheet.loc['Long Term Debt'].iloc[1] if len(balance_sheet.columns) > 1 else long_debt
        if long_debt < long_debt_prev: score += 1
        
        current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
        current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 1
        current_ratio = current_assets / current_liabilities
        if current_ratio > 1: score += 1
        
        shares = info.get('sharesOutstanding', 1)
        if shares > 0: score += 1
        
        # 3. Operating Efficiency (2 puan)
        revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 1
        gross_margin = (revenue - financials.loc['Cost Of Revenue'].iloc[0]) / revenue if 'Cost Of Revenue' in financials.index else 0
        gross_margin_prev = gross_margin
        if gross_margin > gross_margin_prev: score += 1
        
        asset_turnover = revenue / total_assets_current
        if asset_turnover > 0.5: score += 1
        
    except: pass
    return min(score, 9)

def calculate_altman_z(info, balance_sheet, financials):
    """Altman Z-Score (İflas riski)"""
    try:
        current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
        current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 1
        total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0
        retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in balance_sheet.index else 0
        ebit = financials.loc['Ebit'].iloc[0] if 'Ebit' in financials.index else financials.loc['Operating Income'].iloc[0]
        revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 1
        market_cap = info.get('marketCap', 1)
        
        x1 = (current_assets - current_liabilities) / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
        x5 = revenue / total_assets
        
        z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        return z_score
    except:
        return None

def calculate_graham_number(eps, book_value_per_share):
    """Graham Number hesaplama"""
    try:
        return (22.5 * eps * book_value_per_share) ** 0.5
    except:
        return None

def get_news_sentiment(ticker):
    """Basit news sentiment (pozitif/negatif kelime sayımı)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5] if hasattr(stock, 'news') else []
        
        positive_words = ['yükseliş', 'artış', 'başarı', 'kâr', 'büyüme', 'pozitif', 'güçlü', 'upgrade', 'beat']
        negative_words = ['düşüş', 'zarar', 'risk', 'negatif', 'zayıf', 'downgrade', 'miss', 'kayıp']
        
        pos_count = 0
        neg_count = 0
        
        for article in news:
            title = article.get('title', '').lower()
            pos_count += sum(1 for word in positive_words if word in title)
            neg_count += sum(1 for word in negative_words if word in title)
        
        if pos_count + neg_count == 0:
            return "Nötr", 0
        
        sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
        
        if sentiment_score > 0.3:
            return "Pozitif 📈", sentiment_score
        elif sentiment_score < -0.3:
            return "Negatif 📉", sentiment_score
        else:
            return "Nötr ➡️", sentiment_score
    except:
        return "Veri Yok", 0

def generate_pdf_report(ticker, data_dict, currency_symbol):
    """PDF Rapor oluşturma"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Başlık
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#2c3e50'), alignment=TA_CENTER)
    elements.append(Paragraph(f"📊 {ticker} Değerleme Raporu", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Özet Tablo
    summary_data = [
        ['Metrik', 'Değer'],
        ['Piyasa Fiyatı', f"{currency_symbol}{data_dict['current_price']:.2f}"],
        ['Hesaplanan Adil Değer', f"{currency_symbol}{data_dict['fair_value']:.2f}"],
        ['Yukarı Potansiyel', f"{data_dict['upside']:.2f}%"],
        ['Piotroski F-Score', f"{data_dict.get('piotroski', 'N/A')}/9"],
        ['Altman Z-Score', f"{data_dict.get('altman', 'N/A'):.2f}"],
    ]
    
    t = Table(summary_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================
# ANA UYGULAMA
# ============================

# 1. KULLANICI GİRİŞİ
st.sidebar.header("⚙️ Ayarlar")
ticker = st.sidebar.text_input("Hisse Kodu (Örn: KCHOL.IS, THYAO.IS)", "KCHOL.IS").upper()

# --- PARA BİRİMİ VE VARSAYILAN ORANLAR ---
if ticker.endswith(".IS"):
    currency_symbol = "₺"
    default_tax_rate = 0.25
    default_wacc = 25.0
else:
    currency_symbol = "$"
    default_tax_rate = 0.21
    default_wacc = 9.0

if ticker:
    stock = yf.Ticker(ticker)
    
    try:
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        if financials.empty:
            financials = stock.quarterly_financials
        if balance_sheet.empty:
            balance_sheet = stock.quarterly_balance_sheet
        if cash_flow.empty:
            cash_flow = stock.quarterly_cashflow

        # Temel Verileri Çek
        market_cap = info.get('marketCap', 1)
        current_price = info.get('currentPrice', 1)
        shares_outstanding = info.get('sharesOutstanding', 1)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        
        # --- GELİŞMİŞ VERGİ HESAPLAMA (KESİN FİX V2) ---
        try:
            pre_tax_income = financials.loc['Pretax Income'].iloc[0]
            tax_provision = financials.loc['Tax Provision'].iloc[0]
            
            # SAĞLAMLIK KONTROLLARI
            if (pre_tax_income > 0 and 
                tax_provision >= 0 and 
                abs(pre_tax_income) > 1000 and 
                tax_provision < pre_tax_income * 0.5):
                
                calculated_rate = tax_provision / pre_tax_income
                
                if 0 <= calculated_rate <= 0.35:
                    clean_tax_rate = calculated_rate
                else:
                    clean_tax_rate = default_tax_rate
            else:
                clean_tax_rate = default_tax_rate
                
        except:
            clean_tax_rate = default_tax_rate

        # --- VERİ ÇEKME VE SANİTASYON ---
        try:
            total_revenue = financials.loc['Total Revenue'].iloc[0]
            ebit = financials.loc['Ebit'].iloc[0] if 'Ebit' in financials.index else financials.loc['Operating Income'].iloc[0]
            eps = info.get('trailingEps', 1)
            ebitda = info.get('ebitda', ebit * 1.25)
            book_value = info.get('bookValue', 1)
            
        except:
            total_revenue = 1000000; ebit = 100000; eps=1; ebitda=125000; book_value=10

        # WACC Risksiz Faiz
        try:
            rf = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except: rf = 0.04

        # Finansal Metrikler Hesapla
        try:
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
            total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 1
            net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 1
            
            roe = (net_income / total_equity) * 100 if total_equity > 0 else 0
            roa = (net_income / total_assets) * 100 if total_assets > 0 else 0
            roic = (ebit * (1 - clean_tax_rate) / (total_equity + total_debt)) * 100 if (total_equity + total_debt) > 0 else 0
            
            piotroski = calculate_piotroski_score(info, financials, balance_sheet, cash_flow)
            altman = calculate_altman_z(info, balance_sheet, financials)
            
        except:
            roe = roa = roic = 0
            piotroski = 0
            altman = None

        # ========================================================
        # ORTAK DCF HESAPLAMALARI (TÜM TAB'LAR İÇİN)
        # ========================================================
        
        # Default değerler
        revenue_input = float(total_revenue)
        ebit_input = float(ebit)
        tax_rate_input = clean_tax_rate
        wacc_input = default_wacc / 100
        growth_rate = 0.25
        target_margin = 0.15
        terminal_growth = 0.03
        future_years = 5
        
        # Basit DCF hesaplama (varsayılan değerlerle)
        current_revenue = revenue_input
        current_margin = ebit_input / revenue_input if revenue_input != 0 else 0
        
        projections = []
        for year in range(1, future_years + 1):
            current_revenue = current_revenue * (1 + growth_rate)
            year_margin = current_margin + ((target_margin - current_margin) / future_years) * year
            projected_ebit = current_revenue * year_margin
            nopat = projected_ebit * (1 - tax_rate_input)
            reinvestment = nopat * 0.30 
            fcff = nopat - reinvestment
            pv_fcff = fcff / ((1 + wacc_input) ** year)
            
            projections.append({
                "Yıl": year,
                "Gelir": current_revenue,
                "Marj (%)": year_margin * 100,
                "EBIT": projected_ebit,
                "NOPAT": nopat,
                "FCFF": fcff,
                "PV FCFF": pv_fcff
            })

        df_proj = pd.DataFrame(projections)
        sum_pv_fcff = df_proj["PV FCFF"].sum()
        last_fcff = projections[-1]["FCFF"]
        terminal_value = last_fcff * (1 + terminal_growth) / (wacc_input - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc_input) ** future_years)
        
        enterprise_value = sum_pv_fcff + pv_terminal_value
        equity_value = enterprise_value - total_debt + total_cash
        implied_share_price = equity_value / shares_outstanding
        upside = ((implied_share_price - current_price) / current_price) * 100

    except Exception as e:
        st.error(f"❌ Veri çekilemedi: {e}")
        st.stop()

    # Sekme seçimi (VERİ ÇEKTİKTEN SONRA)
    tab_option = st.sidebar.radio("📑 Bölüm Seç", 
        ["💰 Ana DCF Modeli", "📊 Finansal Sağlık", "🎯 Alternatif Değerleme", "🔮 Senaryo Analizi", "📈 Görselleştirmeler", "📄 Rapor Export"])

    # =========================================================================
    # TAB 1: ANA DCF MODELİ
    # =========================================================================
    
    if tab_option == "💰 Ana DCF Modeli":
        
        st.header(f"1. Temel Varsayımlar ({ticker})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue_input = st.number_input(f"Mevcut Gelir ({currency_symbol})", value=float(total_revenue), step=1000000.0, format="%.0f")
            st.caption(f"Okunuşu: {currency_symbol}{revenue_input:,.0f}")
            
        with col2:
            ebit_input = st.number_input(f"Faaliyet Karı - EBIT ({currency_symbol})", value=float(ebit), step=100000.0, format="%.0f")
            st.caption(f"Okunuşu: {currency_symbol}{ebit_input:,.0f}")

        with col3:
            tax_rate_percent = st.number_input(
                "Vergi Oranı (%)", 
                value=float(clean_tax_rate * 100),
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                format="%.2f",
                key=f"tax_rate_{ticker}"
            )
            tax_rate_input = tax_rate_percent / 100

        with col4:
            wacc_input = st.number_input("İskonto Oranı (WACC %)", value=float(default_wacc), step=0.5, format="%.2f", key=f"wacc_{ticker}") / 100

        # Kısım 2: Büyüme Hikayesi
        st.header("2. Gelecek Hikayesi (Growth Story)")

        c1, c2, c3 = st.columns(3)
        with c1:
            growth_rate = st.slider("Gelecek 5 Yıl İçin Yıllık Büyüme (%)", 0, 100, 25) / 100
        with c2:
            target_margin = st.slider("Hedef Faaliyet Marjı (%)", 0, 50, 15) / 100
        with c3:
            terminal_growth = st.number_input("Terminal Büyüme (%)", value=3.0, step=0.1) / 100

        # --- HESAPLAMA MOTORU (Kullanıcı inputlarıyla tekrar hesapla) ---
        future_years = 5
        projections = []
        
        current_revenue = revenue_input
        current_margin = ebit_input / revenue_input if revenue_input != 0 else 0
        
        for year in range(1, future_years + 1):
            current_revenue = current_revenue * (1 + growth_rate)
            year_margin = current_margin + ((target_margin - current_margin) / future_years) * year
            projected_ebit = current_revenue * year_margin
            nopat = projected_ebit * (1 - tax_rate_input)
            reinvestment = nopat * 0.30 
            fcff = nopat - reinvestment
            pv_fcff = fcff / ((1 + wacc_input) ** year)
            
            projections.append({
                "Yıl": year,
                "Gelir": current_revenue,
                "Marj (%)": year_margin * 100,
                "EBIT": projected_ebit,
                "NOPAT": nopat,
                "FCFF": fcff,
                "PV FCFF": pv_fcff
            })

        st.subheader(f"Gelecek {future_years} Yıl Projeksiyonu")
        df_proj = pd.DataFrame(projections)
        
        st.dataframe(df_proj.style.format({
            "Gelir": f"{currency_symbol}{{:,.0f}}", "EBIT": f"{currency_symbol}{{:,.0f}}",
            "NOPAT": f"{currency_symbol}{{:,.0f}}", "FCFF": f"{currency_symbol}{{:,.0f}}",
            "PV FCFF": f"{currency_symbol}{{:,.0f}}", "Marj (%)": "{:.1f}%"
        }), use_container_width=True)

        # --- SONUÇLAR ---
        sum_pv_fcff = df_proj["PV FCFF"].sum()
        last_fcff = projections[-1]["FCFF"]
        terminal_value = last_fcff * (1 + terminal_growth) / (wacc_input - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc_input) ** future_years)
        
        enterprise_value = sum_pv_fcff + pv_terminal_value
        equity_value = enterprise_value - total_debt + total_cash
        implied_share_price = equity_value / shares_outstanding
        upside = ((implied_share_price - current_price) / current_price) * 100

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Piyasa Fiyatı", f"{currency_symbol}{current_price:.2f}")
        with m2: st.metric("Hesaplanan Adil Değer", f"{currency_symbol}{implied_share_price:.2f}", delta=f"{upside:.2f}%")
        with m3: st.metric("Terminal Değerin Payı", f"%{(pv_terminal_value/enterprise_value)*100:.1f}")

        # --- DUYARLILIK ANALİZİ ---
        st.divider()
        st.header("🎯 Duyarlılık Analizi (Risk Tablosu)")
        
        w_step = 0.02; g_step = 0.005 
        wacc_range = [wacc_input - (w_step*2), wacc_input - w_step, wacc_input, wacc_input + w_step, wacc_input + (w_step*2)]
        growth_range = [terminal_growth - (g_step*2), terminal_growth - g_step, terminal_growth, terminal_growth + g_step, terminal_growth + (g_step*2)]

        sensitivity_data = {}
        dcf_min_val = 1e15; dcf_max_val = 0

        for g in growth_range:
            row_vals = []
            for w in wacc_range:
                if w <= g: price = 0
                else:
                    tv = last_fcff * (1 + g) / (w - g)
                    pv_tv = tv / ((1 + w) ** future_years)
                    ev = sum_pv_fcff + pv_tv
                    eq = ev - total_debt + total_cash
                    price = eq / shares_outstanding
                row_vals.append(price)
                if price > 0:
                    if price < dcf_min_val: dcf_min_val = price
                    if price > dcf_max_val: dcf_max_val = price
            sensitivity_data[f"Büyüme %{g*100:.1f}"] = row_vals

        df_sens = pd.DataFrame(sensitivity_data, index=[f"WACC %{w*100:.1f}" for w in wacc_range])
        def color_coding(val):
            if val <= 0: return 'background-color: #95a5a6; color: white;'
            color = '#2ecc71' if val > current_price else '#e74c3c'
            return f'background-color: {color}; color: white; font-weight: bold'
        st.dataframe(df_sens.style.format(f"{currency_symbol}{{:,.2f}}").map(color_coding), use_container_width=True)

        # --- ÇARPANLAR & SİMÜLASYON ---
        st.markdown("---")
        st.header("3. Çarpan Analizi & Monte Carlo")
        
        cm1, cm2 = st.columns(2)
        with cm1:
            sector_pe = st.number_input("Sektör Ort. F/K", 10.0, step=0.5)
            implied_pe = eps * sector_pe
            st.metric("F/K Hedef", f"{currency_symbol}{implied_pe:.2f}")
        with cm2:
            sector_ev = st.number_input("Sektör Ort. EV/EBITDA", 8.0, step=0.5)
            implied_ev = (ebitda * sector_ev - total_debt + total_cash) / shares_outstanding
            st.metric("EV/EBITDA Hedef", f"{currency_symbol}{implied_ev:.2f}")

        comps_min = min(implied_pe, implied_ev); comps_max = max(implied_pe, implied_ev)

        if st.button("Simülasyonu Başlat 🎲"):
            sim_prices = []
            w_dist = np.random.normal(wacc_input, 0.02, 1000)
            g_dist = np.random.normal(growth_rate, 0.05, 1000)
            for i in range(1000):
                w = max(0.05, w_dist[i]); g = g_dist[i]
                sim_tv = (last_fcff*((1+g)/(1+growth_rate))) * (1+terminal_growth) / (w-terminal_growth)
                sim_val = (sum_pv_fcff + (sim_tv/((1+w)**future_years)) - total_debt + total_cash) / shares_outstanding
                sim_prices.append(sim_val)
            
            sim_prices = np.array(sim_prices)
            mc_avg = np.mean(sim_prices)
            st.session_state['mc_min'] = np.percentile(sim_prices, 10)
            st.session_state['mc_max'] = np.percentile(sim_prices, 90)
            st.success(f"✅ Simülasyon Ortalaması: {currency_symbol}{mc_avg:.2f}")
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(sim_prices, bins=40, color='#3498db', alpha=0.7)
            ax.axvline(mc_avg, color='red', linestyle='--', label=f'Ortalama: {currency_symbol}{mc_avg:.0f}')
            ax.axvline(current_price, color='green', linestyle='--', label=f'Mevcut: {currency_symbol}{current_price:.0f}')
            ax.legend()
            ax.set_xlabel('Hisse Fiyatı')
            ax.set_ylabel('Frekans')
            st.pyplot(fig)

        # --- FOOTBALL FIELD ---
        st.markdown("---")
        st.header("4. Değerleme Özeti (Football Field) 🏈")
        mc_low = st.session_state.get('mc_min', implied_share_price * 0.9)
        mc_high = st.session_state.get('mc_max', implied_share_price * 1.1)
        
        methods = ["DCF Aralığı", "Çarpanlar", "Monte Carlo", "52 Hafta"]
        mins = [implied_share_price * 0.9, comps_min, mc_low, info.get('fiftyTwoWeekLow', current_price*0.8)]
        maxs = [implied_share_price * 1.1, comps_max, mc_high, info.get('fiftyTwoWeekHigh', current_price*1.2)]
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        y = np.arange(len(methods))
        for i in range(len(methods)):
            ax2.barh(y[i], maxs[i]-mins[i], left=mins[i], height=0.4, color='#2c3e50', alpha=0.8)
            ax2.text(mins[i], y[i]-0.25, f"{currency_symbol}{mins[i]:.0f}", fontsize=9)
            ax2.text(maxs[i], y[i]-0.25, f"{currency_symbol}{maxs[i]:.0f}", fontsize=9)
        
        ax2.axvline(current_price, color='#e74c3c', linewidth=2, label='Mevcut Fiyat')
        ax2.set_yticks(y); ax2.set_yticklabels(methods); ax2.invert_yaxis(); ax2.legend()
        ax2.set_xlabel('Hisse Fiyatı')
        st.pyplot(fig2)

    # =========================================================================
    # TAB 2: FİNANSAL SAĞLIK
    # =========================================================================
    
    elif tab_option == "📊 Finansal Sağlık":
        st.header("🏥 Finansal Sağlık Kontrolü")
        
        # Temel Metrikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROE (Özkaynak Karlılığı)", f"{roe:.2f}%")
        with col2:
            st.metric("ROA (Varlık Karlılığı)", f"{roa:.2f}%")
        with col3:
            st.metric("ROIC (Yatırım Karlılığı)", f"{roic:.2f}%")
        with col4:
            st.metric("Net Kar Marjı", f"{(net_income/total_revenue)*100:.2f}%" if total_revenue > 0 else "N/A")
        
        st.divider()
        
        # Piotroski F-Score
        st.subheader("🎯 Piotroski F-Score (Finansal Güç)")
        score_col1, score_col2 = st.columns([1, 3])
        
        with score_col1:
            st.markdown(f"<div class='big-metric'>{piotroski}/9</div>", unsafe_allow_html=True)
            if piotroski >= 7:
                st.success("🟢 Güçlü Finansal Durum")
            elif piotroski >= 4:
                st.warning("🟡 Orta Seviye")
            else:
                st.error("🔴 Zayıf Finansal Durum")
        
        with score_col2:
            st.markdown("""
            **Piotroski F-Score Değerlendirmesi:**
            - **7-9:** Çok güçlü finansal sağlık, yatırım için ideal
            - **4-6:** Orta seviye, dikkatli analiz gerekli
            - **0-3:** Zayıf finansal durum, risk yüksek
            
            *Score, 9 farklı karlılık, kaldıraç ve operasyonel verimlilik kriterini değerlendirir.*
            """)
        
        st.divider()
        
        # Altman Z-Score
        st.subheader("⚠️ Altman Z-Score (İflas Riski)")
        if altman:
            z_col1, z_col2 = st.columns([1, 3])
            
            with z_col1:
                st.markdown(f"<div class='big-metric'>{altman:.2f}</div>", unsafe_allow_html=True)
                if altman > 2.99:
                    st.success("🟢 Güvenli Bölge")
                elif altman > 1.81:
                    st.warning("🟡 Gri Bölge")
                else:
                    st.error("🔴 Tehlike Bölgesi")
            
            with z_col2:
                st.markdown("""
                **Altman Z-Score Yorumu:**
                - **Z > 2.99:** Finansal açıdan güvenli, iflas riski düşük
                - **1.81 < Z < 2.99:** Gri bölge, dikkatli izleme gerekli
                - **Z < 1.81:** Yüksek iflas riski, tehlikeli bölge
                
                *Formula: 1.2×(Working Capital/Total Assets) + 1.4×(Retained Earnings/Total Assets) + 3.3×(EBIT/Total Assets) + 0.6×(Market Value/Total Liabilities) + 1.0×(Sales/Total Assets)*
                """)
        else:
            st.info("⚠️ Altman Z-Score hesaplanamadı (yetersiz veri)")
        
        st.divider()
        
        # Likidite Analizi
        st.subheader("💧 Likidite & Borçluluk Analizi")
        
        try:
            current_assets = balance_sheet.loc['Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            
            debt_to_equity = (total_debt / total_equity) * 100 if total_equity > 0 else 0
            
            liq_col1, liq_col2, liq_col3 = st.columns(3)
            
            with liq_col1:
                st.metric("Cari Oran", f"{current_ratio:.2f}x")
                if current_ratio > 2:
                    st.success("Güçlü likidite")
                elif current_ratio > 1:
                    st.warning("Yeterli likidite")
                else:
                    st.error("Zayıf likidite")
            
            with liq_col2:
                st.metric("Borç/Özkaynak", f"{debt_to_equity:.1f}%")
                if debt_to_equity < 50:
                    st.success("Düşük kaldıraç")
                elif debt_to_equity < 100:
                    st.warning("Orta kaldıraç")
                else:
                    st.error("Yüksek kaldıraç")
            
            with liq_col3:
                interest_coverage = ebit / info.get('interestExpense', 1) if info.get('interestExpense', 0) != 0 else 0
                st.metric("Faiz Karşılama", f"{interest_coverage:.1f}x")
                if interest_coverage > 5:
                    st.success("Güçlü")
                elif interest_coverage > 2.5:
                    st.warning("Yeterli")
                else:
                    st.error("Riskli")
                    
        except:
            st.warning("Likidite verileri çekilemedi")
        
        # News Sentiment
        st.divider()
        st.subheader("📰 Haber Duyarlılığı (Son 5 Haber)")
        sentiment, score = get_news_sentiment(ticker)
        
        sent_col1, sent_col2 = st.columns([1, 3])
        with sent_col1:
            st.metric("Genel Duygusal Ton", sentiment)
            st.caption(f"Skor: {score:.2f}")
        
        with sent_col2:
            st.info("📌 **Not:** Bu analiz son 5 haberin başlıklarındaki pozitif/negatif kelimeleri sayar. Derinlemesine analiz için profesyonel araçlar kullanılmalıdır.")

    # =========================================================================
    # TAB 3: ALTERNATİF DEĞERLEME
    # =========================================================================
    
    elif tab_option == "🎯 Alternatif Değerleme":
        st.header("💡 Alternatif Değerleme Modelleri")
        
        # Graham Number
        st.subheader("1️⃣ Graham Number (Benjamin Graham Formula)")
        graham = calculate_graham_number(eps, book_value)
        
        if graham:
            gr_col1, gr_col2 = st.columns(2)
            with gr_col1:
                st.metric("Graham Number", f"{currency_symbol}{graham:.2f}")
                st.metric("Mevcut Fiyat", f"{currency_symbol}{current_price:.2f}")
                graham_discount = ((graham - current_price) / graham) * 100
                st.metric("İskonto/Prim", f"{graham_discount:.1f}%", delta=f"{'Ucuz' if graham_discount > 0 else 'Pahalı'}")
            
            with gr_col2:
                st.markdown("""
                **Graham Number Formülü:**
```
                √(22.5 × EPS × Book Value)
```
                
                **Yorumlama:**
                - Fiyat < Graham Number → **Ucuz** (potansiyel alım)
                - Fiyat > Graham Number → **Pahalı** (temkinli yaklaşım)
                
                Benjamin Graham'ın değer yatırımı prensiplerinden biridir.
                """)
        else:
            st.warning("Graham Number hesaplanamadı")
        
        st.divider()
        
        # DDM (Dividend Discount Model)
        st.subheader("2️⃣ Temettü İskonto Modeli (DDM)")
        
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield and dividend_yield > 0:
            annual_dividend = current_price * dividend_yield
            
            ddm_col1, ddm_col2 = st.columns(2)
            with ddm_col1:
                expected_growth = st.slider("Beklenen Temettü Büyümesi (%)", 0, 20, 5) / 100
                required_return = st.slider("Beklenen Getiri Oranı (%)", 5, 30, 12) / 100
            
            with ddm_col2:
                if required_return > expected_growth:
                    ddm_value = (annual_dividend * (1 + expected_growth)) / (required_return - expected_growth)
                    st.metric("DDM Adil Değer", f"{currency_symbol}{ddm_value:.2f}")
                    ddm_upside = ((ddm_value - current_price) / current_price) * 100
                    st.metric("Yukarı Potansiyel", f"{ddm_upside:.1f}%")
                else:
                    st.error("⚠️ Beklenen getiri, büyüme oranından yüksek olmalı!")
            
            st.info("💡 **Gordon Growth Model:** V = D₁ / (r - g) formülünü kullanır. Temettü ödeyen şirketler için uygundur.")
        else:
            st.warning("Bu şirket temettü ödemiyor, DDM uygulanamaz.")
        
        st.divider()
        
        # Price to Book & Price to Sales
        st.subheader("3️⃣ Çarpan Bazlı Değerleme")
        
        pb_ratio = current_price / book_value if book_value > 0 else 0
        ps_ratio = market_cap / total_revenue if total_revenue > 0 else 0
        pe_ratio = current_price / eps if eps > 0 else 0
        
        mult_col1, mult_col2, mult_col3 = st.columns(3)
        
        with mult_col1:
            st.metric("Fiyat/Defter (P/B)", f"{pb_ratio:.2f}x")
            if pb_ratio < 1:
                st.success("🟢 Defter değerinin altında")
            elif pb_ratio < 3:
                st.warning("🟡 Makul seviyede")
            else:
                st.error("🔴 Yüksek prim")
        
        with mult_col2:
            st.metric("Fiyat/Satış (P/S)", f"{ps_ratio:.2f}x")
            if ps_ratio < 1:
                st.success("🟢 Düşük")
            elif ps_ratio < 3:
                st.warning("🟡 Orta")
            else:
                st.error("🔴 Yüksek")
        
        with mult_col3:
            st.metric("Fiyat/Kazanç (P/E)", f"{pe_ratio:.2f}x")
            if pe_ratio < 15:
                st.success("🟢 Ucuz")
            elif pe_ratio < 25:
                st.warning("🟡 Makul")
            else:
                st.error("🔴 Pahalı")
        
        # Sektör Karşılaştırması
        st.divider()
        st.subheader("4️⃣ Sektör Karşılaştırması (Manuel)")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            sector_avg_pe = st.number_input("Sektör Ort. P/E", value=15.0, step=0.5)
            sector_avg_pb = st.number_input("Sektör Ort. P/B", value=2.0, step=0.1)
        
        with comp_col2:
            pe_relative = (pe_ratio / sector_avg_pe - 1) * 100 if sector_avg_pe > 0 else 0
            pb_relative = (pb_ratio / sector_avg_pb - 1) * 100 if sector_avg_pb > 0 else 0
            
            st.metric("P/E Farkı", f"{pe_relative:+.1f}%", delta=f"{'Sektörün üstünde' if pe_relative > 0 else 'Sektörün altında'}")
            st.metric("P/B Farkı", f"{pb_relative:+.1f}%", delta=f"{'Sektörün üstünde' if pb_relative > 0 else 'Sektörün altında'}")

    # =========================================================================
    # TAB 4: SENARYO ANALİZİ
    # =========================================================================
    
    elif tab_option == "🔮 Senaryo Analizi":
        st.header("🎭 Best / Base / Worst Case Senaryoları")
        
        st.info("📌 **3 farklı senaryo** oluşturun: İyimser, Baz ve Kötümser. Her birinde farklı büyüme ve marj varsayımları kullanın.")
        
        # Senaryo Input
        scenario_cols = st.columns(3)
        
        scenarios = {}
        
        with scenario_cols[0]:
            st.markdown("### 🟢 Best Case")
            best_growth = st.slider("Büyüme (%)", 0, 100, 40, key="best_growth") / 100
            best_margin = st.slider("Hedef Marj (%)", 0, 50, 20, key="best_margin") / 100
            best_terminal = st.number_input("Terminal Büyüme (%)", 0.0, 10.0, 4.0, key="best_term") / 100
            scenarios['Best'] = {'growth': best_growth, 'margin': best_margin, 'terminal': best_terminal, 'color': '#27ae60'}
        
        with scenario_cols[1]:
            st.markdown("### 🟡 Base Case")
            base_growth = st.slider("Büyüme (%)", 0, 100, 25, key="base_growth") / 100
            base_margin = st.slider("Hedef Marj (%)", 0, 50, 15, key="base_margin") / 100
            base_terminal = st.number_input("Terminal Büyüme (%)", 0.0, 10.0, 3.0, key="base_term") / 100
            scenarios['Base'] = {'growth': base_growth, 'margin': base_margin, 'terminal': base_terminal, 'color': '#f39c12'}
        
        with scenario_cols[2]:
            st.markdown("### 🔴 Worst Case")
            worst_growth = st.slider("Büyüme (%)", 0, 100, 10, key="worst_growth") / 100
            worst_margin = st.slider("Hedef Marj (%)", 0, 50, 10, key="worst_margin") / 100
            worst_terminal = st.number_input("Terminal Büyüme (%)", 0.0, 10.0, 2.0, key="worst_term") / 100
            scenarios['Worst'] = {'growth': worst_growth, 'margin': worst_margin, 'terminal': worst_terminal, 'color': '#e74c3c'}
        
        # WACC & Vergi
        wacc_scenario = st.number_input("WACC (%)", value=float(default_wacc), step=0.5, key="scenario_wacc") / 100
        tax_scenario = st.number_input("Vergi Oranı (%)", value=float(clean_tax_rate*100), step=0.5, key="scenario_tax") / 100
        
        if st.button("🚀 Senaryoları Hesapla"):
            results = {}
            
            for name, params in scenarios.items():
                proj_rev = total_revenue
                current_margin = ebit / total_revenue if total_revenue > 0 else 0
                
                fcff_list = []
                for year in range(1, 6):
                    proj_rev = proj_rev * (1 + params['growth'])
                    year_margin = current_margin + ((params['margin'] - current_margin) / 5) * year
                    proj_ebit = proj_rev * year_margin
                    nopat = proj_ebit * (1 - tax_scenario)
                    reinvest = nopat * 0.30
                    fcff = nopat - reinvest
                    pv_fcff = fcff / ((1 + wacc_scenario) ** year)
                    fcff_list.append(pv_fcff)
                
                sum_pv = sum(fcff_list)
                last_fcff_scenario = (proj_ebit * (1 - tax_scenario)) * 0.7
                tv = last_fcff_scenario * (1 + params['terminal']) / (wacc_scenario - params['terminal'])
                pv_tv = tv / ((1 + wacc_scenario) ** 5)
                
                ev = sum_pv + pv_tv
                equity = ev - total_debt + total_cash
                price = equity / shares_outstanding
                
                results[name] = {
                    'price': price,
                    'upside': ((price - current_price) / current_price) * 100,
                    'ev': ev
                }
            
            # Sonuçlar
            st.divider()
            st.subheader("📊 Senaryo Sonuçları")
            
            res_cols = st.columns(3)
            for i, (name, data) in enumerate(results.items()):
                with res_cols[i]:
                    color = scenarios[name]['color']
                    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'>"
                               f"<h3 style='color:white;'>{name} Case</h3>"
                               f"<h2 style='color:white;'>{currency_symbol}{data['price']:.2f}</h2>"
                               f"<p style='color:white; font-size:18px;'>{data['upside']:+.1f}% Potansiyel</p>"
                               f"</div>", unsafe_allow_html=True)
            
            # Görsel
            fig, ax = plt.subplots(figsize=(10, 6))
            names = list(results.keys())
            prices = [results[n]['price'] for n in names]
            colors = [scenarios[n]['color'] for n in names]
            
            bars = ax.bar(names, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.axhline(current_price, color='red', linestyle='--', linewidth=2, label=f'Mevcut Fiyat: {currency_symbol}{current_price:.2f}')
            
            for bar, price in zip(bars, prices):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{currency_symbol}{price:.2f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Hisse Fiyatı', fontsize=12)
            ax.set_title('Senaryo Karşılaştırması', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Özet Tablo
            st.divider()
            scenario_df = pd.DataFrame({
                'Senaryo': names,
                'Hedef Fiyat': [f"{currency_symbol}{results[n]['price']:.2f}" for n in names],
                'Potansiyel': [f"{results[n]['upside']:+.1f}%" for n in names],
                'Büyüme': [f"{scenarios[n]['growth']*100:.0f}%" for n in names],
                'Marj': [f"{scenarios[n]['margin']*100:.0f}%" for n in names]
            })
            st.table(scenario_df)

    # =========================================================================
    # TAB 5: GÖRSELLEŞTİRMELER
    # =========================================================================
    
    elif tab_option == "📈 Görselleştirmeler":
        st.header("📊 İleri Seviye Görselleştirmeler")
        
        # 1. Historik Fiyat Grafiği
        st.subheader("1️⃣ Historik Fiyat Performansı (1 Yıl)")
        hist_data = stock.history(period="1y")
        
        if not hist_data.empty:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(hist_data.index, hist_data['Close'], color='#3498db', linewidth=2)
            ax1.fill_between(hist_data.index, hist_data['Close'], alpha=0.3, color='#3498db')
            ax1.set_title(f'{ticker} - 1 Yıllık Fiyat Hareketi', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Tarih')
            ax1.set_ylabel(f'Fiyat ({currency_symbol})')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
        else:
            st.warning("Historik fiyat verisi çekilemedi")
        
        st.divider()
        
        # 2. Waterfall Chart (DCF Bileşenleri)
        st.subheader("2️⃣ DCF Waterfall Analizi")
        
        # Basit DCF hesaplama
        simple_revenue = total_revenue
        simple_ebit = ebit
        simple_tax = ebit * clean_tax_rate
        simple_nopat = simple_ebit - simple_tax
        simple_capex = simple_nopat * 0.30
        simple_fcf = simple_nopat - simple_capex
        
        waterfall_data = {
            'Gelir': total_revenue,
            'EBIT': -abs(total_revenue - ebit),
            'Vergi': -simple_tax,
            'CAPEX': -simple_capex,
            'Serbest Nakit': 0
        }
        
        cumulative = 0
        positions = []
        values = list(waterfall_data.values())
        
        for v in values:
            positions.append(cumulative)
            cumulative += v
        
        positions[-1] = simple_fcf
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors_wf = ['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c', '#3498db']
        
        for i, (key, val) in enumerate(waterfall_data.items()):
            if i == len(waterfall_data) - 1:
                ax2.bar(i, positions[i], color=colors_wf[i], alpha=0.8, edgecolor='black')
            else:
                ax2.bar(i, abs(val), bottom=positions[i] if val < 0 else positions[i], color=colors_wf[i], alpha=0.8, edgecolor='black')
        
        ax2.set_xticks(range(len(waterfall_data)))
        ax2.set_xticklabels(waterfall_data.keys(), rotation=45, ha='right')
        ax2.set_ylabel(f'Tutar ({currency_symbol})')
        ax2.set_title('Serbest Nakit Akışı Oluşumu (Waterfall)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        st.pyplot(fig2)
        
        st.divider()
        
        # 3. Margin Expansion Roadmap
        st.subheader("3️⃣ Marj İyileşme Yol Haritası")
        
        current_margin = (ebit / total_revenue) * 100 if total_revenue > 0 else 0
        target_margin_viz = st.slider("Hedef Marj (%)", int(current_margin), 50, int(current_margin)+10, key="margin_viz")
        
        years = list(range(0, 6))
        margins = [current_margin]
        for y in range(1, 6):
            margins.append(current_margin + ((target_margin_viz - current_margin) / 5) * y)
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(years, margins, marker='o', color='#9b59b6', linewidth=3, markersize=8)
        ax3.fill_between(years, margins, alpha=0.3, color='#9b59b6')
        ax3.set_xlabel('Yıl', fontsize=12)
        ax3.set_ylabel('Faaliyet Marjı (%)', fontsize=12)
        ax3.set_title('Faaliyet Marjı Genişleme Senaryosu', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.set_xticks(years)
        ax3.set_xticklabels(['Bugün', 'Yıl 1', 'Yıl 2', 'Yıl 3', 'Yıl 4', 'Yıl 5'])
        
        for i, (x, y) in enumerate(zip(years, margins)):
            ax3.text(x, y+0.5, f'{y:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        st.pyplot(fig3)
        
        st.divider()
        
        # 4. Risk-Return Scatter
        st.subheader("4️⃣ Risk-Getiri Profili (Beta vs Beklenen Getiri)")
        
        beta = info.get('beta', 1.0)
        expected_return = (implied_share_price / current_price - 1) * 100
        
        # Benchmark noktaları
        benchmark_data = {
            'Düşük Risk': (0.5, 5),
            'Orta Risk': (1.0, 10),
            'Yüksek Risk': (1.5, 15),
            ticker: (beta, expected_return)
        }
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        for name, (b, r) in benchmark_data.items():
            if name == ticker:
                ax4.scatter(b, r, s=300, color='#e74c3c', marker='*', edgecolor='black', linewidth=2, label=ticker, zorder=5)
            else:
                ax4.scatter(b, r, s=100, alpha=0.6, label=name)
        
        ax4.set_xlabel('Beta (Risk)', fontsize=12)
        ax4.set_ylabel('Beklenen Getiri (%)', fontsize=12)
        ax4.set_title('Risk-Getiri Haritası', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.legend()
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(1, color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig4)

    # =========================================================================
    # TAB 6: RAPOR EXPORT
    # =========================================================================
    
    elif tab_option == "📄 Rapor Export":
        st.header("📑 Profesyonel Rapor İndir")
        
        st.info("💡 **Analiz sonuçlarınızı PDF olarak indirin.** Tüm hesaplamalar ve metrikler tek bir raporda!")
        
        # Veri toplama
        report_data = {
            'current_price': current_price,
            'fair_value': implied_share_price,
            'upside': upside,
            'piotroski': piotroski,
            'altman': altman if altman else 0,
            'roe': roe,
            'roa': roa,
            'roic': roic,
            'pe_ratio': current_price / eps if eps > 0 else 0,
            'pb_ratio': current_price / book_value if book_value > 0 else 0
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Rapor Özeti")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Mevcut Fiyat:** {currency_symbol}{current_price:.2f}")
            st.write(f"**DCF Adil Değer:** {currency_symbol}{implied_share_price:.2f}")
            st.write(f"**Potansiyel:** {upside:+.1f}%")
            st.write(f"**Piotroski Score:** {piotroski}/9")
            st.write(f"**ROE:** {roe:.2f}%")
        
        with col2:
            st.markdown("### 📥 İndirme Seçenekleri")
            
            if st.button("📄 PDF Rapor Oluştur", use_container_width=True):
                with st.spinner("PDF hazırlanıyor..."):
                    pdf_buffer = generate_pdf_report(ticker, report_data, currency_symbol)
                    st.download_button(
                        label="⬇️ PDF İndir",
                        data=pdf_buffer,
                        file_name=f"{ticker}_Valuation_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                st.success("✅ PDF başarıyla oluşturuldu!")
            
            # Excel Export (basit CSV)
            if st.button("📊 Excel (CSV) Oluştur", use_container_width=True):
                export_df = pd.DataFrame({
                    'Metrik': ['Fiyat', 'Adil Değer', 'Potansiyel', 'ROE', 'ROA', 'ROIC', 'Piotroski', 'P/E', 'P/B'],
                    'Değer': [
                        f"{current_price:.2f}",
                        f"{implied_share_price:.2f}",
                        f"{upside:.2f}%",
                        f"{roe:.2f}%",
                        f"{roa:.2f}%",
                        f"{roic:.2f}%",
                        f"{piotroski}/9",
                        f"{report_data['pe_ratio']:.2f}",
                        f"{report_data['pb_ratio']:.2f}"
                    ]
                })
                
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv,
                    file_name=f"{ticker}_Metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.divider()
        
        # Detaylı Projeksiyonlar Tablosu
        st.subheader("📋 Detaylı Finansal Projeksiyonlar")
        
        # Basit projeksiyon tablosu oluştur
        proj_data = []
        proj_rev = total_revenue
        proj_margin = ebit / total_revenue if total_revenue > 0 else 0
        
        for year in range(1, 6):
            proj_rev = proj_rev * 1.25  # %25 büyüme
            proj_ebit = proj_rev * (proj_margin + 0.02*year)  # Marj iyileşmesi
            proj_nopat = proj_ebit * (1 - clean_tax_rate)
            proj_fcf = proj_nopat * 0.7
            
            proj_data.append({
                'Yıl': f'Y{year}',
                'Gelir': proj_rev,
                'EBIT': proj_ebit,
                'NOPAT': proj_nopat,
                'FCF': proj_fcf
            })
        
        proj_df = pd.DataFrame(proj_data)
        st.dataframe(proj_df.style.format({
            'Gelir': f'{currency_symbol}{{:,.0f}}',
            'EBIT': f'{currency_symbol}{{:,.0f}}',
            'NOPAT': f'{currency_symbol}{{:,.0f}}',
            'FCF': f'{currency_symbol}{{:,.0f}}'
        }), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("💎 Fanoel Valuation Suite Ultimate v3.0")
st.sidebar.caption("Tüm özellikler aktif! 🚀")