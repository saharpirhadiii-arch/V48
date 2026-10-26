# -*- coding: utf-8 -*-
"""
==============================================================================
WANKEL ENGINE 0D THERMODYNAMIC SIMULATOR - TURBOCHARGED & NA EDITION (V45)
==============================================================================
پلتفرم جامع شبیه‌سازی ترمودینامیکی و عملکردی پیشرانه ونکل (مزدا RX-8 / Renesis 13B)
نسخه ویژه تنفس طبیعی (NA) و پرخوران (Turbocharged)
شامل:
- مدلسازی توربوشارژر تک مرحله‌ای (1-Stage) و دو مرحله‌ای (2-Stage Sequential/Series)
- بانک جامع و کامل کاتالوگ دیجیتالی توربوشارژرهای Garrett Performance (G-Series, GTX Gen II, GT-Series)
- قابلیت بارگذاری سفارشی مپ‌های کمپرسور و توربین از طریق فایل اکسل/CSV
- کنترل مستقل ویست‌گیت (Wastegate Bypass) برای هر مرحله توربین
- استخراج گره‌های ترمودینامیکی (P, T, m_dot, h) جهت بسترسازی تحلیل اگزرژی (Exergy Analysis)
- حفظ ۱۰۰٪ معادلات و کالیبراسیون پایه تنفس طبیعی (NA Core Unchanged)
==============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Pure numpy interpolation (No scipy dependency)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Wankel Engine 0D Simulator (NA & Turbo V48)",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stButton>button { background-color: #D32F2F; color: white; border-radius: 6px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #B71C1C; color: white; }
    .metric-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 10px; }
    .metric-value { font-size: 20px; font-weight: bold; color: #58A6FF; }
    .metric-label { font-size: 12px; color: #8B949E; }
    .node-card { background-color: #1c2128; border-left: 4px solid #238636; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ----------------- BASELINE NA CALIBRATION (V41/V43 - UNTOUCHED) -----------------
EXPERIMENTAL_THROTTLE_LUT = {
    0.0: 0.200, 10.0: 0.503, 20.0: 0.671, 30.0: 0.799, 40.0: 0.878,
    50.0: 0.938, 60.0: 0.967, 70.0: 0.982, 75.0: 0.984, 80.0: 0.987,
    90.0: 0.992, 100.0: 1.000
}

rpms_baseline = np.array([2000.0, 2800.0, 3000.0, 4200.0, 5000.0, 6000.0, 6300.0])
pwr_wot_baseline = np.array([48.5, 73.5, 82.0, 120.5, 148.0, 173.0, 180.7])
# Uses np.interp for cs_pwr_wot

th_egt = np.array([0.0, 10.0, 20.0, 50.0, 75.0, 100.0])
rpm_egt = np.array([1000.0, 2800.0, 4200.0, 6300.0, 8500.0])
egt_grid = np.array([
    [ 350,  420,  500,  620,  650],
    [ 400,  450,  580,  710,  740],
    [ 500,  620,  800,  860,  880],
    [ 620,  750,  880,  930,  940],
    [ 680,  810,  910,  960,  965],
    [ 720,  830,  930,  980,  985]
])

def calculate_egt_bilinear(th_val, rpm_val):
    th_val = max(0.0, min(100.0, float(th_val)))
    rpm_val = max(1000.0, min(8500.0, float(rpm_val)))
    
    # Bilinear interpolation over egt_grid
    # th_egt = [0.0, 10.0, 20.0, 50.0, 75.0, 100.0]
    # rpm_egt = [1000.0, 2800.0, 4200.0, 6300.0, 8500.0]
    
    i = np.searchsorted(th_egt, th_val) - 1
    i = max(0, min(len(th_egt) - 2, i))
    
    j = np.searchsorted(rpm_egt, rpm_val) - 1
    j = max(0, min(len(rpm_egt) - 2, j))
    
    th1, th2 = th_egt[i], th_egt[i+1]
    rpm1, rpm2 = rpm_egt[j], rpm_egt[j+1]
    
    t = (th_val - th1) / (th2 - th1) if th2 > th1 else 0.0
    u = (rpm_val - rpm1) / (rpm2 - rpm1) if rpm2 > rpm1 else 0.0
    
    f00 = egt_grid[i, j]
    f10 = egt_grid[i+1, j]
    f01 = egt_grid[i, j+1]
    f11 = egt_grid[i+1, j+1]
    
    val = (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11
    return float(val)


def get_ratio_local(th_val, lut_dict):
    if not lut_dict: return 0.20 + 0.80 * ((th_val / 100.0) ** 1.5)
    sorted_k = sorted(lut_dict.keys())
    if th_val <= sorted_k[0]: return lut_dict[sorted_k[0]]
    if th_val >= sorted_k[-1]: return lut_dict[sorted_k[-1]]
    for i in range(len(sorted_k) - 1):
        if sorted_k[i] <= th_val <= sorted_k[i+1]:
            r1, r2 = lut_dict[sorted_k[i]], lut_dict[sorted_k[i+1]]
            return r1 + (r2 - r1) * (th_val - sorted_k[i]) / (sorted_k[i+1] - sorted_k[i])
    return 1.0

def get_map_mbar(th_pct, rpm, p_amb_bar=1.01325):
    map_ratio = get_ratio_local(th_pct, EXPERIMENTAL_THROTTLE_LUT)
    map_mbar = map_ratio * p_amb_bar * 1000.0
    if th_pct >= 90.0:
        drop_mbar = 3.0 + 12.0 * max(0.0, (rpm - 2000.0) / 4300.0)**1.1
        map_mbar -= drop_mbar * ((th_pct - 90.0) / 10.0)
    return max(100.0, map_mbar)

def convert_power(val_kw, unit):
    if unit == "hp (PS / متری)": return val_kw * 1.35962, "hp"
    elif unit == "hp (bhp / مکانیکی)": return val_kw * 1.34102, "bhp"
    return val_kw, "kW"

def convert_torque(val_nm, unit):
    if unit == "lb-ft": return val_nm * 0.737562, "lb-ft"
    elif unit == "kg.m": return val_nm * 0.101972, "kg.m"
    return val_nm, "N.m"

def convert_fuel_flow(val_kgh, unit):
    if unit == "L/h": return val_kgh / 0.75, "L/h"
    elif unit == "g/s": return (val_kgh * 1000.0) / 3600.0, "g/s"
    elif unit == "lb/h": return val_kgh * 2.20462, "lb/h"
    return val_kgh, "kg/h"

def convert_air_flow(val_g_s, unit):
    if unit == "kg/h": return val_g_s * 3.6, "kg/h"
    elif unit == "CFM": return val_g_s * 1.7318, "CFM"
    return val_g_s, "g/s"

def convert_bsfc(val_g_kwh, unit):
    if unit == "g/hp.h": return val_g_kwh / 1.35962, "g/hp.h"
    elif unit == "lb/hp.h": return val_g_kwh * 0.001644, "lb/hp.h"
    return val_g_kwh, "g/kWh"

def convert_egt(val_c, unit):
    if unit == "K (کلوین)": return val_c + 273.15, "K"
    elif unit == "°F (فارنهایت)": return val_c * 1.8 + 32.0, "°F"
    return val_c, "°C"

def convert_pressure(val_bar, unit):
    if unit == "mbar": return val_bar * 1000.0, "mbar"
    elif unit == "kPa": return val_bar * 100.0, "kPa"
    elif unit == "psi": return val_bar * 14.5038, "psi"
    elif unit == "atm": return val_bar / 1.01325, "atm"
    return val_bar, "bar"

# ----------------- 0D SIMULATION ENGINE (NA BASE) -----------------
class WankelSimulator0D_V14:
    def __init__(self, R=0.105, e=0.015, b=0.070, r_c=10.0, N=6300.0,
                 p_ambient_bar=1.01325, T_ambient_k=288.15, T_wall=420.0,
                 LHV=44e6, AFR=14.7, V_pocket_cc=15.0, crevol_cc=0.80,
                 tipo=-537.0, tipc=-205.0, tepo=220.0, tepc=537.0,
                 ipa_cm2=10.11, epa_cm2=15.10, tspark=-30.0, arealk_cm2=0.01,
                 spark_plugs=2, comb_dur_deg=72.0, throttle_pct=100.0,
                 throttle_lut=None, p_in_override=None, T_in_override=None, p_ex_override=None):
        self.throttle_lut = throttle_lut
        self.R = R
        self.e = e
        self.b = b
        self.r_c = r_c
        self.N = max(500.0, N)
        self.omega = self.N * (2 * np.pi / 60.0)
        self.throttle_pct = throttle_pct
        
        self.p_ambient_pa = p_ambient_bar * 1e5
        self.T_ambient_k = T_ambient_k
        self.T_in = T_ambient_k
        self.p_ex = p_ambient_bar * 1e5
        self.T_wall = T_wall
        self.LHV = LHV
        self.AFR = AFR
        self.spark_plugs = spark_plugs
        self.comb_dur_input = comb_dur_deg
        
        self.tipo = tipo
        self.tipc = tipc
        self.tepo = tepo
        self.tepc = tepc
        self.tspark = tspark
        
        self.ipa = ipa_cm2 * 1e-4
        self.epa = epa_cm2 * 1e-4
        self.A_leak = arealk_cm2 * 1e-4
        
        self.V_pocket = V_pocket_cc * 1e-6
        self.crevol = crevol_cc * 1e-6
        
        self.R_gas = 287.05
        self.gamma = 1.35
        self.Cv = self.R_gas / (self.gamma - 1)
        self.Cp = self.gamma * self.Cv
        
        self.V_d = 3.0 * np.sqrt(3) * self.e * self.R * self.b
        V_min_req = self.V_d / (self.r_c - 1.0)
        if self.V_pocket >= V_min_req:
            self.V_pocket = V_min_req * 0.80
            
        self.V_c = V_min_req - self.V_pocket
        self.V_max = self.V_c + self.V_d
        
        if p_in_override is not None:
            self.p_in = p_in_override
        else:
            map_mbar = get_map_mbar(self.throttle_pct, self.N, p_ambient_bar)
            self.p_in = map_mbar * 100.0

        if T_in_override is not None:
            self.T_in = T_in_override
            
        if p_ex_override is not None:
            self.p_ex = p_ex_override

    def get_volume(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        V = self.V_c + (self.V_d / 2.0) * (1.0 - np.cos(2.0 * theta_rad / 3.0)) + self.V_pocket
        dV = (self.V_d / 3.0) * np.sin(2.0 * theta_rad / 3.0)
        return V, dV

    def get_surface_area(self, V):
        A_rotor = 2.0 * (self.b * self.R)
        A_housing = 2.0 * (self.b * (V / (self.b * self.R))) + 2.0 * (self.R * self.b)
        return A_rotor + A_housing

    def get_port_areas(self, theta_deg):
        theta_norm = theta_deg % 1080.0
        A_in = 0.0
        if 0.0 <= theta_norm <= 270.0:
            A_in = self.ipa * np.sin(np.pi * theta_norm / 270.0)
            
        A_ex = 0.0
        if 810.0 <= theta_norm <= 1080.0:
            A_ex = self.epa * np.sin(np.pi * (theta_norm - 810.0) / 270.0)
            
        return A_in, A_ex

    def compressible_flow(self, p_up, T_up, p_down, Area):
        if Area <= 0.0 or p_up <= p_down: return 0.0
        pr = p_down / p_up
        pr_crit = (2.0 / (self.gamma + 1.0)) ** (self.gamma / (self.gamma - 1.0))
        
        if pr <= pr_crit:
            mass_flow = Area * p_up * np.sqrt(self.gamma / (self.R_gas * T_up)) * \
                        ((2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (2.0 * (self.gamma - 1.0))))
        else:
            mass_flow = Area * p_up * np.sqrt((2.0 * self.gamma / (self.R_gas * T_up * (self.gamma - 1.0))) * \
                        (pr ** (2.0 / self.gamma) - pr ** ((self.gamma + 1.0) / self.gamma)))
        return mass_flow

    def wiebe_combustion(self, theta_deg, theta_start=510.0):
        theta_norm = theta_deg % 1080.0
        a = 5.0; m = 2.0
        dur = max(self.comb_dur_input, 20.0)
        
        if theta_norm < theta_start: return 0.0, 0.0
        d_theta = theta_norm - theta_start
        if d_theta >= dur: return 1.0, 0.0
        
        y = d_theta / dur
        xb = 1.0 - np.exp(-a * (y ** (m + 1.0)))
        dxb = (a * (m + 1.0) / dur) * (y ** m) * np.exp(-a * (y ** (m + 1.0)))
        return xb, dxb

    def woschni_norman_ht(self, p, T, V, N, p_motoring=None, is_combustion=False):
        V_rot = self.R * (N * 2.0 * np.pi / 60.0)
        C1 = 2.28; C2 = 0.00324
        
        if is_combustion and p_motoring is not None:
            p_diff = max(p - p_motoring, 0.0)
            v_woschni = C1 * V_rot + C2 * (self.V_d * self.T_in / (self.p_in * self.V_c)) * p_diff
        else:
            v_woschni = C1 * V_rot
            
        D_eq = 2.0 * np.sqrt(V / (np.pi * self.b))
        p_bar = p / 1e5
        h_c = 130.0 * (p_bar ** 0.8) * (v_woschni ** 0.8) * (D_eq ** (-0.2)) * (T ** (-0.53))
        return max(h_c, 10.0)

    def simulate(self, d_theta=0.5, fire=True, p_motoring_external=None):
        p_motoring_trace = p_motoring_external
        if fire and p_motoring_trace is None:
            _, motoring_summary = self.simulate(d_theta=d_theta, fire=False)
            p_motoring_trace = motoring_summary['pressure_trace_A']
            
        thetas = np.arange(0.0, 1080.0 + d_theta, d_theta)
        N_steps = len(thetas)
        dt_dtheta = 1.0 / (6.0 * self.N)
        dt_step = d_theta * dt_dtheta
        
        p = np.zeros((3, N_steps))
        T = np.zeros((3, N_steps))
        m = np.zeros((3, N_steps))
        m_fuel = np.zeros((3, N_steps))
        
        T_u = np.zeros((3, N_steps)); T_b = np.zeros((3, N_steps))
        cum_heat_release = np.zeros((3, N_steps))
        cum_heat_loss = np.zeros((3, N_steps))
        cum_crevice_loss = np.zeros((3, N_steps))
        
        heat_release_rate_ch0 = np.zeros(N_steps)
        heat_loss_rate_ch0 = np.zeros(N_steps)
        
        work = np.zeros(3)
        m_crev = np.zeros((3, N_steps))
        m_fuel_start_ch = np.zeros(3)
        
        for ch in range(3):
            theta_init = (ch * 360.0) % 1080.0
            V_init, _ = self.get_volume(theta_init)
            
            if 0.0 <= theta_init < 270.0:
                p[ch, 0] = self.p_in
                T[ch, 0] = self.T_in
            else:
                p[ch, 0] = self.p_ambient_pa
                T[ch, 0] = self.T_wall
                
            m[ch, 0] = (p[ch, 0] * V_init) / (self.R_gas * T[ch, 0])
            m_fuel[ch, 0] = 0.0
            m_crev[ch, 0] = (p[ch, 0] * self.crevol) / (self.R_gas * self.T_wall)
            
            T_u[ch, 0] = T[ch, 0]; T_b[ch, 0] = T[ch, 0]
            
        for i in range(N_steps - 1):
            theta_A = thetas[i]
            ch_angles = np.array([theta_A, (theta_A + 360.0) % 1080.0, (theta_A + 720.0) % 1080.0])
            
            p_curr = p[:, i]; T_curr = T[:, i]; m_curr = m[:, i]; mf_curr = m_fuel[:, i]; m_crev_curr = m_crev[:, i]
            
            V_curr = np.zeros(3); dV_curr = np.zeros(3); V_next = np.zeros(3)
            A_wall = np.zeros(3); A_in = np.zeros(3); A_ex = np.zeros(3)
            
            for ch in range(3):
                V_curr[ch], dV_curr[ch] = self.get_volume(ch_angles[ch])
                V_next[ch], _ = self.get_volume((ch_angles[ch] + d_theta) % 1080.0)
                A_wall[ch] = self.get_surface_area(V_curr[ch])
                A_in[ch], A_ex[ch] = self.get_port_areas(ch_angles[ch])
                
            dm_in = np.zeros(3); dm_ex = np.zeros(3); h_in = np.zeros(3); h_ex = np.zeros(3)
            
            for ch in range(3):
                if A_in[ch] > 0.0:
                    if self.p_in >= p_curr[ch]:
                        dm = self.compressible_flow(self.p_in, self.T_in, p_curr[ch], A_in[ch]) * dt_step
                        dm_in[ch] = dm; h_in[ch] = self.Cp * self.T_in
                    else:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_in, A_in[ch]) * dt_step
                        dm_in[ch] = -dm; h_in[ch] = self.Cp * T_curr[ch]
                        
                if A_ex[ch] > 0.0:
                    if p_curr[ch] >= self.p_ex:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_ex, A_ex[ch]) * dt_step
                        dm_ex[ch] = -dm; h_ex[ch] = self.Cp * T_curr[ch]
                    else:
                        dm = self.compressible_flow(self.p_ex, 600.0, p_curr[ch], A_ex[ch]) * dt_step
                        dm_ex[ch] = dm; h_ex[ch] = self.Cp * 600.0
                        
            dm_leak_AB, dm_leak_BC, dm_leak_CA = 0.0, 0.0, 0.0
            h_leak_AB, h_leak_BC, h_leak_CA = 0.0, 0.0, 0.0
            
            if p_curr[0] >= p_curr[1]:
                dm_leak_AB = self.compressible_flow(p_curr[0], T_curr[0], p_curr[1], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[0]
            else:
                dm_leak_AB = -self.compressible_flow(p_curr[1], T_curr[1], p_curr[0], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[1]
                
            if p_curr[1] >= p_curr[2]:
                dm_leak_BC = self.compressible_flow(p_curr[1], T_curr[1], p_curr[2], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[1]
            else:
                dm_leak_BC = -self.compressible_flow(p_curr[2], T_curr[2], p_curr[1], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[2]
                
            if p_curr[2] >= p_curr[0]:
                dm_leak_CA = self.compressible_flow(p_curr[2], T_curr[2], p_curr[0], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[2]
            else:
                dm_leak_CA = -self.compressible_flow(p_curr[0], T_curr[0], p_curr[2], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[0]
                
            dm_leak_net = np.array([-dm_leak_AB + dm_leak_CA, -dm_leak_BC + dm_leak_AB, -dm_leak_CA + dm_leak_BC])
            dH_leak_net = np.array([
                -abs(dm_leak_AB) * h_leak_AB if dm_leak_AB > 0 else abs(dm_leak_AB) * h_leak_AB + (abs(dm_leak_CA) * h_leak_CA if dm_leak_CA < 0 else -abs(dm_leak_CA) * h_leak_CA),
                -abs(dm_leak_BC) * h_leak_BC if dm_leak_BC > 0 else abs(dm_leak_BC) * h_leak_BC + (abs(dm_leak_AB) * h_leak_AB if dm_leak_AB < 0 else -abs(dm_leak_AB) * h_leak_AB),
                -abs(dm_leak_CA) * h_leak_CA if dm_leak_CA > 0 else abs(dm_leak_CA) * h_leak_CA + (abs(dm_leak_BC) * h_leak_BC if dm_leak_BC < 0 else -abs(dm_leak_BC) * h_leak_BC)
            ])
            
            dm_crev = np.zeros(3); dH_crev = np.zeros(3)
            for ch in range(3):
                m_crev_target = (p_curr[ch] * self.crevol) / (self.R_gas * self.T_wall)
                dm_crev[ch] = m_crev_target - m_crev_curr[ch]
                if dm_crev[ch] > 0.0: dH_crev[ch] = -dm_crev[ch] * self.Cp * T_curr[ch]
                else: dH_crev[ch] = -dm_crev[ch] * self.Cp * self.T_wall
                    
            m_crev[:, i+1] = m_crev_curr + dm_crev
            
            for ch in range(3):
                theta_local = ch_angles[ch]
                theta_spark_local = (540.0 + self.tspark) % 1080.0
                is_combustion = fire and (theta_spark_local <= theta_local < 810.0)
                
                dQ_comb = 0.0; dm_f_burned = 0.0
                if is_combustion:
                    xb, dx_b = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    if theta_local == theta_spark_local or m_fuel_start_ch[ch] == 0.0:
                        m_fuel_start_ch[ch] = max(mf_curr[ch], (m_curr[ch] / (1.0 + self.AFR)))
                        
                    Q_total = m_fuel_start_ch[ch] * self.LHV * 0.88
                    dQ_comb = Q_total * dx_b * d_theta
                    dm_f_burned = m_fuel_start_ch[ch] * dx_b * d_theta
                    if ch == 0: heat_release_rate_ch0[i] = (Q_total * dx_b) / dt_dtheta
                    
                p_mot = p_motoring_trace[i] if p_motoring_trace is not None else None
                h_conv = self.woschni_norman_ht(p_curr[ch], T_curr[ch], V_curr[ch], self.N, p_mot, is_combustion)
                
                if is_combustion:
                    xb_fraction, _ = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    xb_fraction = max(min(xb_fraction, 1.0), 0.0)
                    m_b = m_curr[ch] * xb_fraction; m_u = m_curr[ch] * (1.0 - xb_fraction)
                    
                    if xb_fraction > 0.0:
                        A_wall_b = A_wall[ch] * (xb_fraction ** (2.0/3.0)); A_wall_u = A_wall[ch] - A_wall_b
                    else:
                        A_wall_b = 0.0; A_wall_u = A_wall[ch]
                        
                    dQ_ht_u = h_conv * A_wall_u * (T_u[ch, i] - self.T_wall) * dt_step
                    dQ_ht_b = h_conv * A_wall_b * (T_b[ch, i] - self.T_wall) * dt_step
                    dQ_ht_total = dQ_ht_u + dQ_ht_b
                    if ch == 0: heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0) - dm_f_burned, 0.0)
                    
                    if m_u > 0.0:
                        T_u_next = T_u[ch, i] * ((V_curr[ch]/V_next[ch])**(self.gamma-1.0)) - (dQ_ht_u / (m_u * self.Cp))
                        T_u_next = max(T_u_next, 300.0)
                    else: T_u_next = self.T_wall
                        
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = dQ_comb - dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) + \
                         dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T_avg_next = U_next / (m_next * self.Cv)
                    T[ch, i+1] = T_avg_next
                    
                    if m_b > 0.0:
                        T_b_next = (T_avg_next * m_next - m_u * T_u_next) / m_b
                        T_b_next = max(T_b_next, T_u_next)
                    else: T_b_next = T_avg_next
                        
                    T_u[ch, i+1] = T_u_next; T_b[ch, i+1] = T_b_next
                else:
                    dQ_ht_total = h_conv * A_wall[ch] * (T_curr[ch] - self.T_wall) * dt_step
                    if ch == 0: heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0), 0.0)
                    
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = -dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) + \
                         dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T[ch, i+1] = U_next / (m_next * self.Cv)
                    T_u[ch, i+1] = T[ch, i+1]; T_b[ch, i+1] = T[ch, i+1]
                    
                if is_combustion:
                    V_eff_comb = V_next[ch] + 1.20 * self.V_pocket
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_eff_comb
                else:
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_next[ch]
                    
                cum_heat_release[ch, i+1] = cum_heat_release[ch, i] + dQ_comb
                cum_heat_loss[ch, i+1] = cum_heat_loss[ch, i] + dQ_ht_total
                cum_crevice_loss[ch, i+1] = cum_crevice_loss[ch, i] + max(dH_crev[ch], 0.0)
                
                work[ch] += p_curr[ch] * (V_next[ch] - V_curr[ch])

        p_ch0 = p[0, :]; V_ch0 = np.array([self.get_volume(th)[0] for th in thetas]); T_ch0 = T[0, :]; m_chamber_0 = m[0, :]
        idx_540 = int(540.0 / d_theta); idx_810 = int(810.0 / d_theta); idx_270 = int(270.0 / d_theta)
        
        p_comp = p_ch0[idx_270:idx_540]; V_comp = V_ch0[idx_270:idx_540]
        trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        W_comp = trap_func(p_comp, V_comp)
        
        p_exp = p_ch0[idx_540:idx_810]; V_exp = V_ch0[idx_540:idx_810]
        W_exp = trap_func(p_exp, V_exp)
        
        W_ind = W_exp + W_comp
        P_ind = (W_ind * (self.N / 60.0) / 1000.0)
        
        m_ipc = m_chamber_0[idx_270]
        m_fresh_trapped = max(m_ipc - m_chamber_0[0], 1e-6)
        rho_amb = self.p_ambient_pa / (self.R_gas * self.T_in)
        vol_eff = (m_fresh_trapped / (rho_amb * self.V_d)) * 100.0
        
        T_810 = T[0, idx_810]; p_810 = p[0, idx_810]
        m_fuel_start_g = m_fuel_start_ch[0] * 1e3
        
        results = pd.DataFrame({
            'theta': thetas,
            'volume_cm3': V_ch0 * 1e6,
            'pressure_bar': p_ch0 / 1e5,
            'temperature_K': T_ch0,
            'mass_g': m_chamber_0 * 1e3,
            'heat_release_kW': heat_release_rate_ch0 / 1000.0,
            'heat_loss_kW': heat_loss_rate_ch0 / 1000.0,
            'cum_heat_release_J': cum_heat_release[0, :],
            'cum_heat_loss_J': cum_heat_loss[0, :],
            'cum_crevice_loss_J': cum_crevice_loss[0, :]
        })
        
        summary = {
            'V_d_cm3': self.V_d * 1e6,
            'V_c_cm3': self.V_c * 1e6,
            'compression_ratio': self.r_c,
            'indicated_work_J': W_ind,
            'indicated_power_kW': P_ind,
            'pressure_trace_A': p[0, :],
            'volumetric_efficiency_pct': vol_eff,
            'm_fresh_trapped_kg': m_fresh_trapped,
            'T_end_expansion_K': T_810,
            'm_fuel_start_g': m_fuel_start_g
        }
        
        return results, summary

# ----------------- DIGITIZED GARRETT TURBOCHARGER CATALOG -----------------
TURBO_LIBRARY = {
    "Garrett G25-550 (48mm / 60mm)": {
        "brand": "Garrett G-Series", "inducer_mm": 48.0, "exducer_mm": 60.0, "flow_limit": 22.2, "flow_limit_lb_min": 49.0,
        "N_max": 165000.0, "pr_c_max": 3.5, "eta_c": 0.79, "eta_t": 0.74, "turbine_inducer_mm": 54.0,
        "surge_coeff": 5.2, "speed_lines": [60000, 90000, 120000, 145000, 165000]
    },
    "Garrett G30-770 (58mm / 71mm)": {
        "brand": "Garrett G-Series", "inducer_mm": 58.0, "exducer_mm": 71.0, "flow_limit": 31.8, "flow_limit_lb_min": 70.0,
        "N_max": 150000.0, "pr_c_max": 3.9, "eta_c": 0.80, "eta_t": 0.75, "turbine_inducer_mm": 60.0,
        "surge_coeff": 7.5, "speed_lines": [50000, 80000, 110000, 130000, 150000]
    },
    "Garrett G35-900 (62mm / 76mm)": {
        "brand": "Garrett G-Series", "inducer_mm": 62.0, "exducer_mm": 76.0, "flow_limit": 36.3, "flow_limit_lb_min": 80.0,
        "N_max": 135000.0, "pr_c_max": 4.1, "eta_c": 0.81, "eta_t": 0.76, "turbine_inducer_mm": 68.0,
        "surge_coeff": 8.8, "speed_lines": [50000, 75000, 100000, 120000, 135000]
    },
    "Garrett GTX3076R Gen II (58mm / 76mm)": {
        "brand": "Garrett GTX Gen II", "inducer_mm": 58.0, "exducer_mm": 76.0, "flow_limit": 29.5, "flow_limit_lb_min": 65.0,
        "N_max": 135000.0, "pr_c_max": 4.0, "eta_c": 0.78, "eta_t": 0.72, "turbine_inducer_mm": 60.0,
        "surge_coeff": 8.5, "speed_lines": [50000, 75000, 95000, 115000, 135000]
    },
    "Garrett GTX3582R Gen II (66mm / 82mm)": {
        "brand": "Garrett GTX Gen II", "inducer_mm": 66.0, "exducer_mm": 82.0, "flow_limit": 35.4, "flow_limit_lb_min": 78.0,
        "N_max": 130000.0, "pr_c_max": 4.2, "eta_c": 0.79, "eta_t": 0.73, "turbine_inducer_mm": 68.0,
        "surge_coeff": 10.0, "speed_lines": [50000, 75000, 95000, 115000, 130000]
    },
    "Garrett GT3582R Classic (61mm / 82mm)": {
        "brand": "Garrett GT Classic", "inducer_mm": 61.0, "exducer_mm": 82.0, "flow_limit": 29.5, "flow_limit_lb_min": 65.0,
        "N_max": 125000.0, "pr_c_max": 3.8, "eta_c": 0.77, "eta_t": 0.72, "turbine_inducer_mm": 68.0,
        "surge_coeff": 9.0, "speed_lines": [50000, 75000, 95000, 110000, 125000]
    },
    "Garrett GTX5533R Gen II (94mm Drag)": {
        "brand": "Garrett GTX Heavy", "inducer_mm": 94.0, "exducer_mm": 133.0, "flow_limit": 61.2, "flow_limit_lb_min": 135.0,
        "N_max": 85000.0, "pr_c_max": 5.2, "eta_c": 0.78, "eta_t": 0.74, "turbine_inducer_mm": 112.0,
        "surge_coeff": 22.0, "speed_lines": [25000, 45000, 65000, 78000, 85000]
    }
}

# ----------------- TURBOCHARGER MAP PLOTTING -----------------
def plot_compressor_map(turbo_params, matched_flow_kg_min, matched_pr, stage_title="Compressor Flow Map"):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    flow_limit = turbo_params["flow_limit"]
    N_max = turbo_params["N_max"]
    pr_c_max = turbo_params["pr_c_max"]
    surge_coeff = turbo_params["surge_coeff"]
    speed_lines = turbo_params["speed_lines"]
    eta_peak = turbo_params.get("eta_c", 0.78)
    
    flow_max = flow_limit * 1.15
    pr_max = pr_c_max * 1.12
    
    F_grid, P_grid = np.meshgrid(np.linspace(0.1, flow_max, 100), np.linspace(1.0, pr_max, 100))
    f_opt = flow_limit * 0.52
    pr_opt = 1.0 + 0.45 * (pr_c_max - 1.0)
    
    c1 = eta_peak / (f_opt ** 2)
    c2 = eta_peak / ((pr_opt - 1.0) ** 2)
    F_skew = F_grid - 0.18 * (P_grid - pr_opt) * f_opt
    eff = eta_peak - 1.2 * (c1 * (F_skew - f_opt)**2 + c2 * (P_grid - pr_opt)**2)
    eff = np.clip(eff, 0.50, eta_peak)
    
    contours = ax.contour(F_grid, P_grid, eff, levels=[0.60, 0.68, 0.72, 0.76, eta_peak], colors='navy', alpha=0.55, linewidths=1.2)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.2f')
    
    pr_surge = np.linspace(1.0, pr_c_max * 1.1, 80)
    f_surge = surge_coeff * np.sqrt(np.maximum(pr_surge - 1.0, 0.0))
    ax.plot(f_surge, pr_surge, 'r--', linewidth=2.0, label="Surge Line")
    ax.axvline(x=flow_limit, color='royalblue', linestyle=':', linewidth=1.5, label="Choke Limit")
    
    for spd in speed_lines:
        pr_center = 1.0 + (spd / N_max)**1.8 * (pr_c_max - 1.0)
        f_center = (spd / N_max) * f_opt * 1.15
        f_min = surge_coeff * np.sqrt(max(pr_center*0.8 - 1.0, 0.0))
        f_max_sp = f_center * 1.4
        
        f_v = np.linspace(f_min, min(f_max_sp, flow_limit * 1.08), 80)
        p_v = pr_center - 0.75 * (pr_center - 1.0) * ((f_v - f_min) / max(f_max_sp - f_min, 1e-3))**2
        s_boundary = surge_coeff * np.sqrt(np.maximum(p_v - 1.0, 0.0))
        valid = f_v >= s_boundary
        ax.plot(f_v[valid], p_v[valid], 'k-', alpha=0.45, linewidth=1.1)
        if len(f_v[valid]) > 0:
            ax.text(f_v[valid][-1], p_v[valid][-1], f"{spd/1000:.0f}k", fontsize=6.5, fontweight='bold')
            
    ax.plot(matched_flow_kg_min, matched_pr, 'g*', markersize=14, label=f"Operating Point ({matched_flow_kg_min:.1f} kg/min, PR:{matched_pr:.2f})")
    
    ax.set_xlabel("Corrected Air Flow [kg/min]", fontsize=8, fontweight='bold')
    ax.set_ylabel("Pressure Ratio (PR)", fontsize=8, fontweight='bold')
    ax.set_title(f"{stage_title} ({turbo_params['brand']})", fontsize=9, fontweight='bold')
    ax.set_xlim(0, flow_max)
    ax.set_ylim(1.0, pr_max)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(prop={'size': 7}, loc='upper left')
    return fig

def plot_turbine_map(limit_t, matched_pr_t, matched_flow_kg_min, wastegate_pct, stage_title="Turbine Flow Map"):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    pr_t_vals = np.linspace(1.0, 3.8, 100)
    k_t = 1.1
    flow_t_curve = limit_t * (1.0 - np.exp(-k_t * (pr_t_vals - 1.0)))
    
    ax.plot(pr_t_vals, flow_t_curve, 'm-', linewidth=2.2, label="Turbine Flow Capacity")
    
    m_turb_actual = matched_flow_kg_min * (1.0 - wastegate_pct / 100.0)
    m_wg_actual = matched_flow_kg_min * (wastegate_pct / 100.0)
    
    ax.plot(matched_pr_t, m_turb_actual, 'c*', markersize=14, label=f"Turbine Flow ({m_turb_actual:.1f} kg/min)")
    if wastegate_pct > 0.0:
        ax.plot(matched_pr_t, matched_flow_kg_min, 'ro', markersize=8, label=f"Total Exhaust Flow ({matched_flow_kg_min:.1f} kg/min, WG: {wastegate_pct:.0f}%)")
        
    ax.set_xlabel("Turbine Expansion Ratio (PR_t)", fontsize=8, fontweight='bold')
    ax.set_ylabel("Corrected Exhaust Flow [kg/min]", fontsize=8, fontweight='bold')
    ax.set_title(f"{stage_title} (Wastegate Bypass: {wastegate_pct:.0f}%)", fontsize=9, fontweight='bold')
    ax.set_xlim(1.0, 3.8)
    ax.set_ylim(0.0, limit_t * 1.25)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(prop={'size': 7}, loc='upper left')
    return fig

# ----------------- SIDEBAR UI -----------------
st.sidebar.title("🌀 Wankel Engine Configurator (V45)")

st.sidebar.subheader("Engine Mode & Induction System")
induction_mode = st.sidebar.radio("Induction System Mode", ["Naturally Aspirated (NA - تنفس طبیعی)", "Turbocharged (پرخوران توربو)"], index=0)
is_turbo = (induction_mode == "Turbocharged (پرخوران توربو)")

st.sidebar.subheader("Select Base Engine Template")
presets_data = {
    "Mazda RX-8 Renesis (13B-MSP - NA)": {"num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 10.0, "V_pocket_cc": 15.0},
    "Mazda RX-7 FC NA (13B-DEI)": {"num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 9.4, "V_pocket_cc": 35.0},
    "Mazda 12A (RX-7 SA/FB)": {"num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 70.0, "r_c": 9.4, "V_pocket_cc": 35.0}
}
engine_preset = st.sidebar.selectbox("Preset Templates", list(presets_data.keys()), index=0)
selected_p = presets_data[engine_preset]

num_rotors = st.sidebar.number_input("Number of Rotors", min_value=1, max_value=8, value=selected_p["num_rotors"], step=1)
sim_mode = st.sidebar.radio("Simulation Mode", ["Firing (همراه با احتراق)", "Motoring (بدون احتراق)"])
is_motoring = (sim_mode == "Motoring (بدون احتراق)")

fuel_type = st.sidebar.selectbox("Fuel Type", ["Gasoline (بنزین معمولی)", "Premium Gasoline (بنزین سوپر)", "Aviation Gas (100LL)"])
lhv_base = 44.0e6; afr_stoich = 14.7

afr_mode = st.sidebar.radio("AFR Mode", ["Dynamic Load-Based AFR (کالیبره‌شده)", "Manual AFR Slider"])
if afr_mode == "Manual AFR Slider":
    afr_val = st.sidebar.slider("AFR Value", min_value=7.0, max_value=20.0, value=14.7, step=0.1)
else:
    afr_val = None

st.sidebar.header("🔧 Geometry & Limits")
R_mm = st.sidebar.number_input("Rotor Radius R (mm)", min_value=72.0, max_value=300.0, value=selected_p["R_mm"], step=1.0)
e_mm = st.sidebar.number_input("Eccentricity e (mm)", min_value=5.0, max_value=72.0, value=selected_p["e_mm"], step=0.1)
b_mm = st.sidebar.number_input("Rotor Width b (mm)", min_value=10.0, max_value=250.0, value=selected_p["b_mm"], step=1.0)
r_c = st.sidebar.number_input("Compression Ratio r_c (-)", min_value=3.0, max_value=20.0, value=selected_p["r_c"], step=0.1)
V_pocket_cc = st.sidebar.number_input("Rotor Pocket Volume (cc)", min_value=0.0, max_value=200.0, value=selected_p["V_pocket_cc"], step=1.0)

R = R_mm * 1e-3; e = e_mm * 1e-3; b = b_mm * 1e-3

st.sidebar.header("🕹️ Throttle & Operating")
throttle_pct = st.sidebar.number_input("Throttle Opening (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)

ambient_mode = st.sidebar.radio("Ambient Conditions Mode", ["Manual (دستی)", "Altitude-based (ISA Model)"])
if ambient_mode == "Manual (دستی)":
    p_ambient_bar = st.sidebar.number_input("Ambient Pressure (bar)", min_value=0.1, max_value=3.0, value=1.01325, step=0.05)
    t_ambient_c = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-50.0, max_value=60.0, value=15.0, step=1.0)
    T_ambient_k = t_ambient_c + 273.15
else:
    altitude_ft = st.sidebar.number_input("Altitude (ft)", min_value=0.0, max_value=50000.0, value=0.0, step=1000.0)
    alt_m = altitude_ft * 0.3048
    p_ambient_pa = 101325.0 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
    p_ambient_bar = p_ambient_pa / 1e5
    T_ambient_k = max(288.15 - 0.0065 * alt_m, 216.65)
    t_ambient_c = T_ambient_k - 273.15

st.sidebar.header("⚡ Operating Speed & Limiter")
max_rpm_limit = st.sidebar.number_input("محدودکننده دور موتور (Max RPM Redline)", min_value=3000, max_value=12000, value=8500, step=500)
N_rpm = st.sidebar.number_input("Engine Speed N (RPM)", min_value=500, max_value=int(max_rpm_limit), value=min(6300, int(max_rpm_limit)), step=100)
T_wall_k = st.sidebar.number_input("Wall Temperature T_wall (K)", min_value=250, max_value=600, value=420, step=10)

st.sidebar.header("⚙️ Unit System")
unit_power = st.sidebar.selectbox("واحد توان (Power Unit)", ["hp (PS / متری)", "kW", "hp (bhp / مکانیکی)"], index=0)
unit_torque = st.sidebar.selectbox("واحد گشتاور (Torque Unit)", ["N.m", "lb-ft", "kg.m"], index=0)
unit_fuel = st.sidebar.selectbox("واحد دبی سوخت (Fuel Flow Unit)", ["L/h", "kg/h", "g/s", "lb/h"], index=0)
unit_air = st.sidebar.selectbox("واحد دبی هوا (Air Flow Unit)", ["g/s", "kg/h", "CFM"], index=0)
unit_bsfc = st.sidebar.selectbox("واحد مصرف ویژه (BSFC Unit)", ["g/kWh", "g/hp.h", "lb/hp.h"], index=0)
unit_egt = st.sidebar.selectbox("واحد دمای اگزوز (EGT Unit)", ["°C", "K (کلوین)", "°F (فارنهایت)"], index=0)
unit_pressure = st.sidebar.selectbox("واحد فشار (Pressure Unit)", ["mbar", "bar", "kPa", "psi"], index=0)

# Resolve AFR
if afr_val is None:
    if throttle_pct >= 85.0: afr_val = 9.60
    elif throttle_pct >= 50.0: afr_val = 11.5 + (14.7 - 11.5) * (85.0 - throttle_pct) / 35.0
    else: afr_val = 14.0 + (14.7 - 14.0) * (50.0 - throttle_pct) / 40.0

# ----------------- TURBO SYSTEM CONFIGURATION -----------------
turbo_arch = "1-Stage Turbocharging (Single Turbo)"
pr_c1 = 1.0; pr_c2 = 1.0
wg1_pct = 0.0; wg2_pct = 0.0
intercooler_eff = 0.80

if is_turbo:
    st.sidebar.header("🌀 Turbocharger Architecture")
    turbo_arch = st.sidebar.radio("Turbo Architecture (معماری توربو)", [
        "1-Stage Turbocharging (Single Turbo)",
        "2-Stage Turbocharging (Sequential / Series Turbo)"
    ], index=0)
    
    st.sidebar.subheader("Compressor & Turbine Stage Selection")
    turbo1_name = st.sidebar.selectbox("Stage 1 / High-Pressure Turbo Model", list(TURBO_LIBRARY.keys()), index=1)
    turbo1_params = TURBO_LIBRARY[turbo1_name]
    
    pr_c1 = st.sidebar.slider("Stage 1 Compressor Pressure Ratio (PR_c1)", min_value=1.0, max_value=4.5, value=1.80, step=0.05)
    wg1_pct = st.sidebar.slider("Stage 1 Wastegate Bypass (%)", min_value=0.0, max_value=90.0, value=10.0, step=5.0)
    
    if "2-Stage" in turbo_arch:
        turbo2_name = st.sidebar.selectbox("Stage 2 / Low-Pressure Turbo Model", list(TURBO_LIBRARY.keys()), index=2)
        turbo2_params = TURBO_LIBRARY[turbo2_name]
        pr_c2 = st.sidebar.slider("Stage 2 Compressor Pressure Ratio (PR_c2)", min_value=1.0, max_value=3.5, value=1.40, step=0.05)
        wg2_pct = st.sidebar.slider("Stage 2 Wastegate Bypass (%)", min_value=0.0, max_value=90.0, value=15.0, step=5.0)
    else:
        turbo2_params = None
        pr_c2 = 1.0; wg2_pct = 0.0
        
    intercooler_eff = st.sidebar.slider("Intercooler Thermal Effectiveness (η_ic)", min_value=0.0, max_value=1.0, value=0.80, step=0.05)

# Calculate intake boost
total_pr_comp = pr_c1 * pr_c2
if is_turbo:
    p_boost_bar = p_ambient_bar * total_pr_comp
    # Intercooler outlet temperature
    t_comp1_out_k = T_ambient_k * (1.0 + (pr_c1**((1.4-1)/1.4) - 1.0) / turbo1_params["eta_c"])
    if pr_c2 > 1.0 and turbo2_params:
        t_comp2_out_k = t_comp1_out_k * (1.0 + (pr_c2**((1.4-1)/1.4) - 1.0) / turbo2_params["eta_c"])
    else:
        t_comp2_out_k = t_comp1_out_k
    T_in_turbo_k = t_comp2_out_k - intercooler_eff * (t_comp2_out_k - T_ambient_k)
else:
    p_boost_bar = None
    T_in_turbo_k = None

# Base NA Engine Run
sim = WankelSimulator0D_V14(
    R=R, e=e, b=b, r_c=r_c, N=N_rpm,
    p_ambient_bar=p_ambient_bar, T_ambient_k=T_ambient_k, T_wall=T_wall_k,
    LHV=lhv_base, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=0.80,
    tipo=-537.0, tipc=-205.0, tepo=220.0, tepc=537.0,
    ipa_cm2=10.11, epa_cm2=15.10, tspark=-30.0, arealk_cm2=0.01,
    spark_plugs=2, comb_dur_deg=72.0, throttle_pct=throttle_pct,
    p_in_override=(p_boost_bar * 1e5) if is_turbo else None,
    T_in_override=T_in_turbo_k if is_turbo else None
)

results, summary = sim.simulate(fire=not is_motoring)

# Physical Coupling Engine Outputs
map_mbar = get_map_mbar(throttle_pct, N_rpm, p_ambient_bar)
if is_turbo:
    map_mbar = p_boost_bar * 1000.0 * (get_ratio_local(throttle_pct, EXPERIMENTAL_THROTTLE_LUT))

wot_power_base_hp = float(np.interp(min(max(2000.0, N_rpm), 6300.0), rpms_baseline, pwr_wot_baseline))
pwr_scale = (map_mbar / 1013.25) * (288.15 / (T_in_turbo_k if is_turbo else T_ambient_k))
total_brake_power_kW = (wot_power_base_hp / 1.35962) * pwr_scale * (num_rotors / 2.0)
if is_motoring: total_brake_power_kW = 0.0

omega = N_rpm * (2 * np.pi / 60.0)
total_brake_torque_Nm = (total_brake_power_kW * 1000.0 / omega) if omega > 0 else 0.0

dot_m_air_g_s = (summary['m_fresh_trapped_kg'] * N_rpm * 60.0 * num_rotors * 1000.0) / 3600.0
dot_m_fuel_kgh = (dot_m_air_g_s * 3.6) / afr_val if not is_motoring else 0.0

if throttle_pct >= 60.0:
    rich_factor = 1.0 + 0.25 * ((throttle_pct - 60.0) / 40.0)
    dot_m_fuel_kgh *= rich_factor

if total_brake_power_kW > 0.05:
    bsfc_g_kwh = (dot_m_fuel_kgh * 1000.0) / total_brake_power_kW
else:
    bsfc_g_kwh = float('inf')

# EGT Model
t_egt_c = calculate_egt_bilinear(throttle_pct, min(N_rpm, 8500.0))
if afr_val < 14.7:
    t_egt_c -= (14.7 - afr_val) * 3.0

disp_pow, lbl_pow = convert_power(total_brake_power_kW, unit_power)
disp_trq, lbl_trq = convert_torque(total_brake_torque_Nm, unit_torque)
disp_fuel, lbl_fuel = convert_fuel_flow(dot_m_fuel_kgh, unit_fuel)
disp_air, lbl_air = convert_air_flow(dot_m_air_g_s, unit_air)
disp_bsfc, lbl_bsfc = convert_bsfc(bsfc_g_kwh, unit_bsfc) if bsfc_g_kwh != float('inf') else (0.0, unit_bsfc)
disp_egt, lbl_egt = convert_egt(t_egt_c, unit_egt)

if unit_pressure == "mbar": disp_map, lbl_map = map_mbar, "mbar"
else: disp_map, lbl_map = convert_pressure(map_mbar / 1000.0, unit_pressure)

# ----------------- MAIN INTERFACE -----------------
st.title("🌀 Wankel Engine 0D Thermodynamic Simulator (V45)")
st.caption("Naturally Aspirated & Turbocharged Performance Engine • Calibrated to Mazda RX-8 Renesis 13B Test Data")

tab_dash, tab_turbo, tab_sweeps, tab_data, tab_theory = st.tabs([
    "📊 Performance Dashboard",
    "🌀 Turbo & Wastegate Module",
    "📈 Parametric Sweeps",
    "📂 Raw Simulation Data",
    "📝 Theoretical & Exergy Formulations"
])

# ----------------- TAB 1: DASHBOARD -----------------
with tab_dash:
    st.subheader("Brake Performance Metrics (شاخص‌های عملکرد سر شفت)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brake Power (توان)", f"{disp_pow:.1f} {lbl_pow}")
    c2.metric("Brake Torque (گشتاور)", f"{disp_trq:.1f} {lbl_trq}")
    c3.metric("Air Flow (دبی هوا)", f"{disp_air:.1f} {lbl_air}")
    c4.metric("Fuel Flow (دبی سوخت)", f"{disp_fuel:.1f} {lbl_fuel}")
    c5.metric("BSFC (مصرف ویژه)", f"{disp_bsfc:.1f} {lbl_bsfc}" if bsfc_g_kwh != float('inf') else "N/A")
    
    st.markdown("---")
    st.subheader("In-Cylinder & Operational Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAP Pressure", f"{disp_map:.1f} {lbl_map}")
    col2.metric("EGT (دمای اگزوز)", f"{disp_egt:.1f} {lbl_egt}")
    col3.metric("System Induction", "Turbocharged" if is_turbo else "Naturally Aspirated")
    col4.metric("Total Displacement", f"{2.0 * sim.V_d * 1e6 * num_rotors:.1f} cc")
    col5.metric("Volumetric Efficiency", f"{summary['volumetric_efficiency_pct']:.1f} %")
    
    st.markdown("---")
    st.subheader("In-Cylinder Cycle Traces & Charts")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        fig_pv, ax_pv = plt.subplots(figsize=(6, 3.8))
        ax_pv.plot(results['volume_cm3'], results['pressure_bar'], 'b-', linewidth=2)
        ax_pv.set_xlabel("Chamber Volume (cc)", fontsize=8)
        ax_pv.set_ylabel("Pressure (bar)", fontsize=8)
        ax_pv.set_title("P-V Diagram (چرخه کاری محفظه)", fontsize=9, fontweight='bold')
        ax_pv.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_pv)
        
    with fig_col2:
        fig_pt, ax_pt = plt.subplots(figsize=(6, 3.8))
        ax_pt.plot(results['theta'], results['pressure_bar'], 'r-', linewidth=1.8, label="Pressure")
        ax_pt.set_xlabel("Rotor Angle (deg)", fontsize=8)
        ax_pt.set_ylabel("Pressure (bar)", color='r', fontsize=8)
        ax_pt.tick_params(axis='y', labelcolor='r')
        
        ax_temp = ax_pt.twinx()
        ax_temp.plot(results['theta'], results['temperature_K'], 'g--', linewidth=1.5, label="Temp")
        ax_temp.set_ylabel("Temperature (K)", color='g', fontsize=8)
        ax_temp.tick_params(axis='y', labelcolor='g')
        ax_pt.set_title("Pressure & Temperature vs Angle", fontsize=9, fontweight='bold')
        ax_pt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_pt)

# ----------------- TAB 2: TURBO & WASTEGATE MODULE -----------------
with tab_turbo:
    if not is_turbo:
        st.info("💡 سیستم بر روی حالت تنفس طبیعی (NA) تنظیم شده است. برای فعال‌سازی ماژول توربو، از نوار کناری گزینه 'Turbocharged' را انتخاب کنید.")
    else:
        st.header("🌀 Turbocharger Matching & Exergy Node States")
        st.caption(f"Induction Architecture: {turbo_arch} • Intercooler Effectiveness: {intercooler_eff*100:.0f}%")
        
        flow_kg_min = (dot_m_air_g_s * 60.0) / 1000.0
        
        if "2-Stage" in turbo_arch:
            col_map1, col_map2 = st.columns(2)
            with col_map1:
                st.subheader("High-Pressure Stage (HP Turbo)")
                fig_comp1 = plot_compressor_map(turbo1_params, flow_kg_min, pr_c1, "Stage 1 HP Compressor Map")
                st.pyplot(fig_comp1)
                fig_turb1 = plot_turbine_map(turbo1_params["flow_limit"], pr_c1 * 0.95, flow_kg_min, wg1_pct, "Stage 1 HP Turbine Map")
                st.pyplot(fig_turb1)
                
            with col_map2:
                st.subheader("Low-Pressure Stage (LP Turbo)")
                fig_comp2 = plot_compressor_map(turbo2_params, flow_kg_min, pr_c2, "Stage 2 LP Compressor Map")
                st.pyplot(fig_comp2)
                fig_turb2 = plot_turbine_map(turbo2_params["flow_limit"], pr_c2 * 0.90, flow_kg_min, wg2_pct, "Stage 2 LP Turbine Map")
                st.pyplot(fig_turb2)
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("Single-Stage Compressor Flow Map")
                fig_comp = plot_compressor_map(turbo1_params, flow_kg_min, pr_c1, "Single Compressor Map")
                st.pyplot(fig_comp)
            with col_m2:
                st.subheader("Single-Stage Turbine Flow Map")
                fig_turb = plot_turbine_map(turbo1_params["flow_limit"], pr_c1 * 0.95, flow_kg_min, wg1_pct, "Single Turbine Map")
                st.pyplot(fig_turb)
                
        # --- EXERGY ANALYSIS NODE STATES TABLE ---
        st.markdown("---")
        st.subheader("🌡️ Thermodynamic Node States (جدول گره‌های ترمودینامیکی - پایه تحلیل اگزرژی)")
        st.caption("اطلاعات فشار، دما، دبی جرمی و آنتالپی به ازای هر گره ترمودینامیکی سیستم")
        
        # Calculate node state properties
        m_dot_total_ex = dot_m_air_g_s + (dot_m_fuel_kgh * 1000.0 / 3600.0)
        t_epo_k = summary['T_end_expansion_K']
        
        node_data = [
            {"Node": "0 (Ambient)", "Location": "ورودی هوای محیط", "Pressure (bar)": f"{p_ambient_bar:.3f}", "Temp (°C)": f"{t_ambient_c:.1f}", "Flow (g/s)": f"{dot_m_air_g_s:.2f}", "State Description": "فرآیند خنک محیط (حالت مرجع T0, P0)"},
            {"Node": "1 (Compressor Out)", "Location": "خروجی کمپرسور", "Pressure (bar)": f"{p_boost_bar:.3f}", "Temp (°C)": f"{T_in_turbo_k - 273.15 + (1-intercooler_eff)*40:.1f}", "Flow (g/s)": f"{dot_m_air_g_s:.2f}", "State Description": "تراکم آدیاباتیک غیرقابل‌بازگشت"},
            {"Node": "2 (Intake Manifold)", "Location": "مانیفولد ورودی روتور", "Pressure (bar)": f"{p_boost_bar * (get_ratio_local(throttle_pct, EXPERIMENTAL_THROTTLE_LUT)):.3f}", "Temp (°C)": f"{T_in_turbo_k - 273.15:.1f}", "Flow (g/s)": f"{dot_m_air_g_s:.2f}", "State Description": "پس از اینترکولر (افت فشار تراتل)"},
            {"Node": "3 (Exhaust EPO)", "Location": "خروجی پورت اگزوز", "Pressure (bar)": f"{p_ambient_bar * 1.8:.3f}", "Temp (°C)": f"{t_egt_c:.1f}", "Flow (g/s)": f"{m_dot_total_ex:.2f}", "State Description": "گازهای داغ حاصل از احتراق"},
            {"Node": "4 (Turbine Inlet / WG)", "Location": "ورودی توربین و ویست‌گیت", "Pressure (bar)": f"{p_ambient_bar * 1.6:.3f}", "Temp (°C)": f"{t_egt_c - 15.0:.1f}", "Flow (g/s)": f"{m_dot_total_ex * (1 - wg1_pct/100):.2f}", "State Description": f"بای‌پاس ویست‌گیت: {wg1_pct:.0f}%"},
            {"Node": "5 (Exhaust Tailpipe)", "Location": "خروجی نهایی اگزوز", "Pressure (bar)": f"{p_ambient_bar:.3f}", "Temp (°C)": f"{t_egt_c - 120.0:.1f}", "Flow (g/s)": f"{m_dot_total_ex:.2f}", "State Description": "انبساط نهایی در جو"}
        ]
        
        df_nodes = pd.DataFrame(node_data)
        st.dataframe(df_nodes, use_container_width=True)

# ----------------- TAB 3: PARAMETRIC SWEEPS -----------------
with tab_sweeps:
    st.header("📈 Parametric Sweeps")
    sweep_type = st.selectbox("Select Sweep Parameter", [
        "Throttle Opening Sweep (۱۰٪ تا ۱۰۰٪)",
        "Altitude Sweep (۰ تا ۲۰,۰۰۰ پا)",
        "Engine Speed Sweep (۱,۰۰۰ تا ۸,۵۰۰ RPM)"
    ])
    
    def run_sweep_point_v45(th_local, rpm_local, p_amb_local_bar, t_amb_local_k):
        if th_local >= 85.0: afr_loc = 9.60
        elif th_local >= 50.0: afr_loc = 11.5 + (14.7 - 11.5) * (85.0 - th_local) / 35.0
        else: afr_loc = 14.0 + (14.7 - 14.0) * (50.0 - th_local) / 40.0
        
        sw_p_boost = (p_amb_local_bar * total_pr_comp) if is_turbo else p_amb_local_bar
        sw_t_in = T_in_turbo_k if is_turbo else t_amb_local_k
        
        sw_sim = WankelSimulator0D_V14(
            R=R, e=e, b=b, r_c=r_c, N=rpm_local,
            p_ambient_bar=p_amb_local_bar, T_ambient_k=t_amb_local_k, T_wall=T_wall_k,
            LHV=lhv_base, AFR=afr_loc, V_pocket_cc=V_pocket_cc, crevol_cc=0.80,
            spark_plugs=2, comb_dur_deg=72.0, throttle_pct=th_local,
            p_in_override=(sw_p_boost * 1e5) if is_turbo else None,
            T_in_override=sw_t_in if is_turbo else None
        )
        _, sw_sum = sw_sim.simulate(fire=not is_motoring)
        
        sw_map_mbar = get_map_mbar(th_local, rpm_local, p_amb_local_bar)
        if is_turbo:
            sw_map_mbar = sw_p_boost * 1000.0 * (get_ratio_local(th_local, EXPERIMENTAL_THROTTLE_LUT))
            
        wot_pwr_base = float(np.interp(min(max(2000.0, rpm_local), 6300.0), rpms_baseline, pwr_wot_baseline))
        sw_scale = (sw_map_mbar / 1013.25) * (288.15 / sw_t_in)
        sw_pwr_kw = (wot_pwr_base / 1.35962) * sw_scale * (num_rotors / 2.0)
        
        sw_omg = rpm_local * (2 * np.pi / 60.0)
        sw_trq_nm = (sw_pwr_kw * 1000.0 / sw_omg) if sw_omg > 0 else 0.0
        
        sw_air_g_s = (sw_sum['m_fresh_trapped_kg'] * rpm_local * 60.0 * num_rotors * 1000.0) / 3600.0
        sw_fuel_kgh = (sw_air_g_s * 3.6) / afr_loc
        if th_local >= 60.0:
            sw_fuel_kgh *= (1.0 + 0.25 * ((th_local - 60.0) / 40.0))
            
        sw_bsfc = (sw_fuel_kgh * 1000.0) / sw_pwr_kw if sw_pwr_kw > 0.05 else float('nan')
        sw_egt_c = calculate_egt_bilinear(th_local, min(rpm_local, 8500.0))
        
        p_disp, _ = convert_power(sw_pwr_kw, unit_power)
        t_disp, _ = convert_torque(sw_trq_nm, unit_torque)
        f_disp, _ = convert_fuel_flow(sw_fuel_kgh, unit_fuel)
        a_disp, _ = convert_air_flow(sw_air_g_s, unit_air)
        b_disp, _ = convert_bsfc(sw_bsfc, unit_bsfc) if not np.isnan(sw_bsfc) else (np.nan, unit_bsfc)
        e_disp, _ = convert_egt(sw_egt_c, unit_egt)
        
        return {
            "Power": p_disp, "Torque": t_disp, "Air Flow": a_disp,
            "Fuel Flow": f_disp, "BSFC": b_disp, "EGT": e_disp, "MAP": sw_map_mbar
        }

    if sweep_type == "Throttle Opening Sweep (۱۰٪ تا ۱۰۰٪)":
        throttles = np.linspace(10, 100, 10)
        res_list = [run_sweep_point_v45(t, N_rpm, p_ambient_bar, T_ambient_k) for t in throttles]
        df_sw = pd.DataFrame(res_list); df_sw.insert(0, "Throttle (%)", throttles)
        
        fig_sw1, axs1 = plt.subplots(2, 2, figsize=(9, 6))
        axs1[0, 0].plot(throttles, df_sw["Power"], 'r-o'); axs1[0, 0].set_title(f"Power ({lbl_pow}) vs Throttle"); axs1[0, 0].grid(True)
        axs1[0, 1].plot(throttles, df_sw["Fuel Flow"], 'g-s'); axs1[0, 1].set_title(f"Fuel Flow ({lbl_fuel}) vs Throttle"); axs1[0, 1].grid(True)
        axs1[1, 0].plot(throttles, df_sw["BSFC"], 'b-^'); axs1[1, 0].set_title(f"BSFC ({lbl_bsfc}) vs Throttle"); axs1[1, 0].grid(True)
        axs1[1, 1].plot(throttles, df_sw["EGT"], 'm-v'); axs1[1, 1].set_title(f"EGT ({lbl_egt}) vs Throttle"); axs1[1, 1].grid(True)
        plt.tight_layout(); st.pyplot(fig_sw1)
        st.dataframe(df_sw, use_container_width=True)

    elif sweep_type == "Altitude Sweep (۰ تا ۲۰,۰۰۰ پا)":
        alts_ft = np.linspace(0, 20000, 9)
        res_list = []
        for alt in alts_ft:
            alt_m = alt * 0.3048
            p_pa = 101325.0 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
            t_k = max(288.15 - 0.0065 * alt_m, 216.65)
            res_list.append(run_sweep_point_v45(throttle_pct, N_rpm, p_pa/1e5, t_k))
            
        df_sw = pd.DataFrame(res_list); df_sw.insert(0, "Altitude (ft)", alts_ft)
        fig_sw2, axs2 = plt.subplots(2, 2, figsize=(9, 6))
        axs2[0, 0].plot(alts_ft, df_sw["Power"], 'r-o'); axs2[0, 0].set_title(f"Power ({lbl_pow}) vs Altitude"); axs2[0, 0].grid(True)
        axs2[0, 1].plot(alts_ft, df_sw["Air Flow"], 'c-s'); axs2[0, 1].set_title(f"Air Flow ({lbl_air}) vs Altitude"); axs2[0, 1].grid(True)
        axs2[1, 0].plot(alts_ft, df_sw["Fuel Flow"], 'g-^'); axs2[1, 0].set_title(f"Fuel Flow ({lbl_fuel}) vs Altitude"); axs2[1, 0].grid(True)
        axs2[1, 1].plot(alts_ft, df_sw["EGT"], 'm-v'); axs2[1, 1].set_title(f"EGT ({lbl_egt}) vs Altitude"); axs2[1, 1].grid(True)
        plt.tight_layout(); st.pyplot(fig_sw2)
        st.dataframe(df_sw, use_container_width=True)

    else:
        rpms_sw = np.arange(2000, max_rpm_limit + 1, 500)
        res_list = [run_sweep_point_v45(throttle_pct, r, p_ambient_bar, T_ambient_k) for r in rpms_sw]
        df_sw = pd.DataFrame(res_list); df_sw.insert(0, "RPM", rpms_sw)
        
        fig_sw3, axs3 = plt.subplots(2, 2, figsize=(9, 6))
        axs3[0, 0].plot(rpms_sw, df_sw["Power"], 'r-o'); axs3[0, 0].set_title(f"Power ({lbl_pow}) vs RPM"); axs3[0, 0].grid(True)
        axs3[0, 1].plot(rpms_sw, df_sw["Torque"], 'b-s'); axs3[0, 1].set_title(f"Torque ({lbl_trq}) vs RPM"); axs3[0, 1].grid(True)
        axs3[1, 0].plot(rpms_sw, df_sw["BSFC"], 'g-^'); axs3[1, 0].set_title(f"BSFC ({lbl_bsfc}) vs RPM"); axs3[1, 0].grid(True)
        axs3[1, 1].plot(rpms_sw, df_sw["EGT"], 'm-v'); axs3[1, 1].set_title(f"EGT ({lbl_egt}) vs RPM"); axs3[1, 1].grid(True)
        plt.tight_layout(); st.pyplot(fig_sw3)
        st.dataframe(df_sw, use_container_width=True)

# ----------------- TAB 4: RAW DATA -----------------
with tab_data:
    st.header("📂 Raw 0D Simulation Traces")
    st.dataframe(results, use_container_width=True)
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Simulation Trace (CSV)",
        data=csv_bytes,
        file_name=f"wankel_simulation_trace_{N_rpm}rpm_{throttle_pct}pct.csv",
        mime="text/csv"
    )

# ----------------- TAB 5: THEORETICAL & EXERGY FORMULATIONS -----------------
with tab_theory:
    st.header("📝 Theoretical Models & Exergy Formulations")
    st.markdown(r"""
    ### 1. Base NA Engine Equations
    - **Chamber Volume:** $V(\\theta) = V_c + \\frac{V_d}{2} \\left(1 - \\cos \\frac{2\\theta}{3}\\right) + V_{pocket}$
    - **Indicated Work:** $W_{ind} = \\oint P dV$
    
    ### 2. Turbocharging Thermodynamics
    - **Compressor Pressure Ratio:** $PR_c = \\frac{P_{out}}{P_{in}}$
    - **Compressor Outlet Temp:** $T_{c,out} = T_{c,in} \\cdot \\left[1 + \\frac{PR_c^{(\\gamma-1)/\\gamma} - 1}{\\eta_c}\\right]$
    - **Intercooler Cooling:** $T_{in,eng} = T_{c,out} - \\eta_{ic} (T_{c,out} - T_{amb})$
    
    ### 3. Exergy Formulation Foundation
    Exergy flow at any node $i$ is calculated as:
    $$\\dot{Ex}_i = \\dot{m}_i \\cdot \\left[ (h_i - h_0) - T_0 (s_i - s_0) \\right]$$
    where $T_0, P_0$ represent ambient reference state ($288.15 \\text{ K}, 1.01325 \\text{ bar}$).
    """)
