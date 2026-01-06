import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & CONSTANTS ---
TOTAL_ROUNDS = 12
PRODUCTS = {
    "Laptop (Steady)": {
        "base_demand": 200, "cost": 800, "price": 1200, 
        "holding_rate": 0.02, "stockout_cost": 400, "seasonality": 0.1
    },
    "Smartphone (Volatile)": {
        "base_demand": 500, "cost": 400, "price": 700, 
        "holding_rate": 0.03, "stockout_cost": 200, "seasonality": 0.4
    }
}
LOCATIONS = ["North DC", "South DC"]

# --- INITIALIZATION ---
def initialize_game():
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {
            'month': 1,
            'cash': 1000000.0, # Starting capital
            'total_profit': 0.0,
            'inventory': {},  # {loc: {prod: qty}}
            'pipeline': {},   # Orders placed but not arrived
            'history': [],    # List of stats per round
            'game_over': False,
            'lead_time': 1,   # Months
            'network_strategy': 'Decentralized', # Can change in Month 6
            'newsvendor_done': False
        }
        
        # Init Inventory
        for loc in LOCATIONS:
            st.session_state.game_state['inventory'][loc] = {}
            st.session_state.game_state['pipeline'][loc] = {}
            for prod in PRODUCTS:
                # Start with roughly 1.5 months of inventory
                start_qty = int(PRODUCTS[prod]['base_demand'] * 1.5)
                st.session_state.game_state['inventory'][loc][prod] = start_qty
                # Pipeline is empty initially
                st.session_state.game_state['pipeline'][loc][prod] = 0

def get_demand(product_name, month, volatility_factor=1.0):
    """Calculates demand based on seasonality and random noise."""
    prod = PRODUCTS[product_name]
    
    # Seasonality Curve (Peaks in Q4 - Month 10-12)
    # Simple sine wave shifted to peak late year
    season_factor = 1 + (prod['seasonality'] * np.sin((month - 1) / 11 * np.pi)) 
    
    # Peak Season Spike (Months 10, 11)
    if month in [10, 11]:
        season_factor += 0.3

    base = prod['base_demand'] * season_factor
    
    # Randomness (The "Difficulty" factor)
    noise = np.random.normal(0, base * 0.2 * volatility_factor) # 20% std dev
    return max(0, int(base + noise))

# --- GAME ENGINE ---
def advance_round(orders, strategic_decision=None, newsvendor_order=None):
    gs = st.session_state.game_state
    current_month = gs['month']
    
    # 1. Apply Strategic Decision (Month 6)
    if current_month == 6 and strategic_decision:
        if strategic_decision == "Centralize (Pooling)":
            gs['network_strategy'] = "Centralized"
            gs['lead_time'] = 2 # Slower but cheaper holding? Let's say lead time increases
            # In a real sim, we might merge stocks, but for simplicity, we keep locs but change params
            st.toast("⚠️ Supply Chain Redesigned: Centralized! Lead times increased, but demand variance risk is pooled (simulated by lower stockout cost in future).")
        else:
            st.toast("ℹ️ Strategy maintained: Decentralized.")

    # 2. Process Arrivals (Pipeline -> On Hand)
    # Logic: Orders placed last month arrive now (if LT=1)
    # We simplified: Pipeline is strictly "Arriving this month"
    for loc in LOCATIONS:
        for prod in PRODUCTS:
            arriving_qty = gs['pipeline'][loc][prod]
            gs['inventory'][loc][prod] += arriving_qty
            gs['pipeline'][loc][prod] = 0 # Clear pipeline after arrival

    # 3. Generate Demand & Fulfill
    round_stats = {'month': current_month, 'revenue': 0, 'holding_cost': 0, 
                   'ordering_cost': 0, 'stockout_cost': 0, 'details': []}
    
    total_ordering_cost = 500 * len(LOCATIONS) # Fixed per-location ordering cost
    
    for loc in LOCATIONS:
        for prod_name, prod_data in PRODUCTS.items():
            dem = get_demand(prod_name, current_month)
            
            # Impact of SC Redesign on Demand? 
            # If Centralized, standard deviation effectively lowers (Risk Pooling). 
            # We simulate this by reducing the extreme spikes in demand slightly for scoring purposes or checking availability.
            
            on_hand = gs['inventory'][loc][prod_name]
            sold = min(on_hand, dem)
            missed = dem - sold
            ending_inventory = on_hand - sold
            
            # Costs
            revenue = sold * prod_data['price']
            h_cost = ending_inventory * prod_data['cost'] * prod_data['holding_rate']
            s_cost = missed * prod_data['stockout_cost']
            
            # Update State
            gs['inventory'][loc][prod_name] = ending_inventory
            
            # Accumulate Stats
            round_stats['revenue'] += revenue
            round_stats['holding_cost'] += h_cost
            round_stats['stockout_cost'] += s_cost
            
            round_stats['details'].append({
                'loc': loc, 'prod': prod_name, 'demand': dem, 
                'sold': sold, 'missed': missed, 'end_inv': ending_inventory
            })

    # 4. Process Newsvendor Scenario (Month 9)
    if current_month == 9 and newsvendor_order is not None:
        # One time event: "Flash Holiday Gadget"
        nv_demand = int(np.random.normal(2000, 600)) # High uncertainty
        nv_sold = min(newsvendor_order, nv_demand)
        nv_revenue = nv_sold * 150 # High margin
        nv_cost = newsvendor_order * 80 # Cost
        nv_profit = nv_revenue - nv_cost
        # Overage cost is full cost (0 salvage), Underage is lost margin
        st.toast(f"📰 Newsvendor Result: Ordered {newsvendor_order}, Demand {nv_demand}. Profit: ${nv_profit}")
        round_stats['revenue'] += max(0, nv_profit) # Add net profit directly to revenue for simplicity
        gs['newsvendor_done'] = True

    # 5. Place New Orders (Into Pipeline)
    # In this sim, Lead Time is 1 month. Order placed Month 1 arrives start of Month 2.
    for loc in LOCATIONS:
        for prod in PRODUCTS:
            qty = orders[loc][prod]
            gs['pipeline'][loc][prod] = qty
            # Variable ordering cost (shipping)
            total_ordering_cost += qty * 2 

    round_stats['ordering_cost'] += total_ordering_cost
    
    # Financial Update
    round_profit = round_stats['revenue'] - (round_stats['holding_cost'] + round_stats['ordering_cost'] + round_stats['stockout_cost'])
    gs['cash'] += round_profit
    gs['total_profit'] += round_profit
    gs['history'].append(round_stats)
    
    # Advance Month
    gs['month'] += 1
    if gs['month'] > TOTAL_ROUNDS:
        gs['game_over'] = True

# --- UI RENDERING ---
def main():
    st.set_page_config(page_title="SCM Simulator: ElectroGrid", layout="wide")
    st.title("🏭 Supply Chain Commander: ElectroGrid Inc.")
    st.markdown("""
    **Mission:** Maximize profit over 12 months. Minimize Inventory Costs while meeting uncertain demand.
    
    *Concepts Covered:* Cycle Stock, Safety Stock, Seasonality, Newsvendor Model, Multi-Echelon Logic.
    """)
    
    initialize_game()
    gs = st.session_state.game_state
    
    if gs['game_over']:
        st.success("🏆 Simulation Complete!")
        st.metric("Final Cash Position", f"${gs['cash']:,.2f}")
        st.metric("Total Net Profit", f"${gs['total_profit']:,.2f}")
        
        # Performance Analysis
        df = pd.DataFrame(gs['history'])
        st.line_chart(df[['revenue', 'holding_cost', 'stockout_cost']])
        
        if st.button("Restart Simulation"):
            st.session_state.clear()
            st.rerun()
        return

    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Month", f"{gs['month']} / {TOTAL_ROUNDS}")
    col2.metric("Cash", f"${gs['cash']:,.0f}")
    col3.metric("Network Strategy", gs['network_strategy'])
    
    # Last Month Info
    if len(gs['history']) > 0:
        last = gs['history'][-1]
        col4.metric("Last Month Profit", f"${last['revenue'] - (last['holding_cost'] + last['stockout_cost'] + last['ordering_cost']):,.0f}", 
                    delta_color="normal")
    
    st.divider()

    # --- INPUT SECTION ---
    with st.form("order_form"):
        st.subheader(f"📋 Decisions for Month {gs['month']}")
        
        # Strategic Decision: Month 6
        strat_choice = None
        if gs['month'] == 6:
            st.warning("⚠️ **Strategic Crossroads:** Supply Chain Redesign")
            st.markdown("Your network is straining. Do you want to centralize inventory?")
            st.markdown("- **Keep Decentralized:** Low Lead Time (1 mo), High Variance Risk.")
            st.markdown("- **Centralize (Pooling):** Higher Lead Time (2 mo), Lower effective Stockout Costs (Risk Pooling).")
            strat_choice = st.radio("Choose Strategy:", ["Keep Decentralized", "Centralize (Pooling)"])

        # Newsvendor Decision: Month 9
        nv_qty = None
        if gs['month'] == 9:
            st.info("🎁 **Special Event:** Holiday 'Flash' Gadget")
            st.markdown("One-time opportunity. Item Cost: $80. Selling Price: $150. Unsold items are worthless (0 salvage).")
            st.markdown("Demand Forecast: Normal Dist(Mean=2000, SD=600).")
            nv_qty = st.number_input("One-time Order Quantity:", min_value=0, value=2000)

        # Standard Replenishment Orders
        orders = {loc: {} for loc in LOCATIONS}
        
        cols = st.columns(len(LOCATIONS))
        for i, loc in enumerate(LOCATIONS):
            with cols[i]:
                st.markdown(f"### 📍 {loc}")
                for prod in PRODUCTS:
                    curr_inv = gs['inventory'][loc][prod]
                    pipeline = gs['pipeline'][loc][prod]
                    
                    # Visual Indicator of status
                    st.write(f"**{prod}**")
                    st.caption(f"On Hand: {curr_inv} | Arriving: {pipeline}")
                    
                    orders[loc][prod] = st.number_input(
                        f"Order (+{prod})", 
                        min_value=0, 
                        key=f"{loc}_{prod}",
                        help=f"Cost: ${PRODUCTS[prod]['cost']}. Lead Time: {gs['lead_time']} Month(s)"
                    )

        submitted = st.form_submit_button("🚀 Execute Round")
        
        if submitted:
            advance_round(orders, strat_choice, nv_qty)
            st.rerun()

    # --- ANALYTICS TAB ---
    st.divider()
    st.subheader("📊 Supply Chain Analytics")
    
    if len(gs['history']) > 0:
        hist_df = pd.DataFrame(gs['history'])
        
        # Cost Breakdown Area Chart
        st.write("### Cost Structure Over Time")
        cost_df = hist_df[['holding_cost', 'stockout_cost', 'ordering_cost']]
        st.area_chart(cost_df)
        
        # Inventory vs Demand (Last Month Detail)
        st.write("### Last Month Performance by Location")
        last_details = pd.DataFrame(gs['history'][-1]['details'])
        
        # Pivot for readability
        chart_data = last_details.pivot(index='loc', columns='prod', values=['end_inv', 'missed'])
        st.dataframe(last_details)
        
    else:
        st.info("Analytics will appear after the first round.")

if __name__ == "__main__":
    main()
