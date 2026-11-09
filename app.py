import streamlit as st
import streamlit.components.v1 as components
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import geopandas as gpd
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
import plotly.express as px

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# RETINASCOPE
# Explainable AI for Diabetic Retinopathy Screening
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_b0_best.pth"
DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

GRADE_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy"
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetinaScope // Aetherial Ocular Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TECHFEST.ORG ("AN AETHERIAL RENAISSANCE") FRONTEND ENGINE
# Multi-Layer Parallax: Farside, Palace Base, Palace Glow,
# Character Sentinel, Transition Fog 1-6, Scroll-Reactive
# Biome Sky Interpolation & 3D Cursor-Tracking Hover Physics
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --tf-void: #02050b;
    --tf-obsidian: #050b16;
    --tf-citadel: #081224;
    --tf-cyan: #00f0ff;
    --tf-cyan-dim: rgba(0, 240, 255, 0.18);
    --tf-cyan-glow: rgba(0, 240, 255, 0.55);
    --tf-gold: #e5c158;
    --tf-gold-bright: #fae084;
    --tf-gold-dim: rgba(229, 193, 88, 0.22);
    --tf-violet: #9d4edd;
    --tf-crimson: #ff3b5c;
    --tf-emerald: #00f59b;
    --tf-amber: #ffaa00;
    --tf-text: #f1f5f9;
    --tf-muted: #94a3b8;
    --tf-scroll-pct: 0;
    --tf-scroll-px: 0px;
    --tf-scroll-vel: 0;
    --tf-sky-top: #02050d;
    --tf-sky-mid: #06152d;
    --tf-sky-bot: #0b2239;
    --tf-nebula-a: rgba(0, 240, 255, 0.16);
    --tf-nebula-b: rgba(229, 193, 88, 0.13);
    --tf-active-accent: #00f0ff;
}

/* ==========================================================
   CSS SCROLL-DRIVEN ANIMATION FALLBACK (NATIVE CHROMIUM)
   ========================================================== */
@keyframes tfScrollBiomeShift {
    0% { background-color: #02050d; }
    25% { background-color: #041226; }
    50% { background-color: #0d061c; }
    75% { background-color: #031317; }
    100% { background-color: #120c04; }
}

@keyframes tfFogDriftOdd {
    0% { transform: translate3d(-5%, 0, 0) scale(1.04); }
    50% { transform: translate3d(5%, -10px, 0) scale(1.09); }
    100% { transform: translate3d(-5%, 0, 0) scale(1.04); }
}

@keyframes tfFogDriftEven {
    0% { transform: translate3d(6%, 0, 0) scale(1.06); }
    50% { transform: translate3d(-6%, 8px, 0) scale(1.02); }
    100% { transform: translate3d(6%, 0, 0) scale(1.06); }
}

@keyframes tfAstrolabeSpin {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}

@keyframes tfPalacePulse {
    0%, 100% { opacity: 0.55; filter: drop-shadow(0 0 18px rgba(0, 240, 255, 0.45)); }
    50% { opacity: 0.96; filter: drop-shadow(0 0 36px rgba(229, 193, 88, 0.78)); }
}

@keyframes tfLaserScan {
    0% { top: -15%; opacity: 0; }
    15% { opacity: 0.9; }
    85% { opacity: 0.9; }
    100% { top: 115%; opacity: 0; }
}

@keyframes tfScrollChevron {
    0%, 100% { transform: translateY(0); opacity: 0.45; }
    50% { transform: translateY(8px); opacity: 1; }
}

/* ==========================================================
   GLOBAL APP SHELL & SCROLL-REACTIVE ATMOSPHERE
   ========================================================== */
.stApp {
    background:
        radial-gradient(circle at 50% calc(12% + var(--tf-scroll-pct, 0) * 0.65%), var(--tf-nebula-a, rgba(0, 240, 255, 0.16)), transparent 45%),
        radial-gradient(circle at calc(82% - var(--tf-scroll-pct, 0) * 0.5%) calc(24% + var(--tf-scroll-pct, 0) * 0.55%), var(--tf-nebula-b, rgba(229, 193, 88, 0.13)), transparent 40%),
        radial-gradient(circle at calc(18% + var(--tf-scroll-pct, 0) * 0.4%) calc(68% - var(--tf-scroll-pct, 0) * 0.35%), rgba(157, 78, 221, 0.14), transparent 45%),
        linear-gradient(180deg, var(--tf-sky-top, #02050d) 0%, var(--tf-sky-mid, #06152d) 52%, var(--tf-sky-bot, #0b2239) 100%) !important;
    background-attachment: fixed !important;
    color: var(--tf-text) !important;
    font-family: 'Rajdhani', sans-serif !important;
    transition: background 0.28s cubic-bezier(0.16, 1, 0.3, 1);
    animation: tfScrollBiomeShift linear both;
    animation-timeline: scroll(root);
    overflow-x: hidden;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 4.8rem !important;
    padding-bottom: 5rem !important;
    position: relative;
    z-index: 5;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 999 !important;
}

/* Custom Cyber Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #02050b;
    border-left: 1px solid rgba(0, 240, 255, 0.12);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00f0ff 0%, #e5c158 100%);
    border-radius: 0px;
}

/* ==========================================================
   FIXED PARALLAX BACKGROUND STAGE (TECHFEST.ORG LAYER STACK)
   Farside -> Palace Base -> Palace Glow -> Character -> Transition Fog 1..6
   ========================================================== */
.tf-fixed-stage {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

/* Layer 0: Farside Horizon & Perspective Grid */
.tf-layer-farside {
    position: absolute;
    inset: -10% 0 0 0;
    background:
        radial-gradient(ellipse 90% 50% at 50% 8%, rgba(0, 240, 255, 0.11), transparent 65%),
        radial-gradient(ellipse 65% 40% at 20% 22%, rgba(229, 193, 88, 0.08), transparent 60%);
    transform: translate3d(0, calc(var(--tf-scroll-px, 0px) * -0.04), 0);
    will-change: transform;
}

.tf-cyber-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 240, 255, 0.042) 1px, transparent 1px),
        linear-gradient(90deg, rgba(229, 193, 88, 0.032) 1px, transparent 1px);
    background-size: 64px 64px;
    mask-image: radial-gradient(ellipse 85% 75% at 50% 45%, #000 25%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse 85% 75% at 50% 45%, #000 25%, transparent 80%);
    transform: translate3d(0, calc(var(--tf-scroll-px, 0px) * -0.09), 0);
    will-change: transform;
}

/* Celestial Astrolabe Rings (Farside Upper Sky) */
.tf-farside-astrolabe {
    position: absolute;
    top: 28%;
    left: 50%;
    width: 880px;
    height: 880px;
    border-radius: 50%;
    border: 1px dashed rgba(229, 193, 88, 0.16);
    box-shadow: inset 0 0 90px rgba(0, 240, 255, 0.05);
    animation: tfAstrolabeSpin 110s linear infinite;
    pointer-events: none;
}

.tf-farside-astrolabe::before {
    content: "";
    position: absolute;
    inset: 68px;
    border-radius: 50%;
    border: 1px solid rgba(0, 240, 255, 0.14);
    border-top-color: rgba(229, 193, 88, 0.55);
    border-bottom-color: rgba(0, 240, 255, 0.55);
}

.tf-farside-astrolabe::after {
    content: "";
    position: absolute;
    inset: 145px;
    border-radius: 50%;
    border: 1px dotted rgba(229, 193, 88, 0.24);
}

/* Layer 2: Palace Base & Palace Glow Silhouette at Bottom Horizon */
.tf-palace-horizon {
    position: absolute;
    bottom: -24px;
    left: 0;
    width: 100%;
    height: 380px;
    pointer-events: none;
    will-change: transform;
}

.tf-palace-glow-svg {
    animation: tfPalacePulse 5.5s ease-in-out infinite;
}

/* Layer 3: Six Transition Fog Layers (Transition Fog 1 - 6) */
.tf-fog-band {
    position: absolute;
    left: -12%;
    width: 124%;
    pointer-events: none;
    mix-blend-mode: screen;
    will-change: transform, opacity;
}

.tf-fog-1 {
    bottom: 0%;
    height: 170px;
    background: radial-gradient(ellipse 65% 80% at 28% 65%, rgba(0, 240, 255, 0.12), transparent 70%),
                radial-gradient(ellipse 60% 75% at 74% 55%, rgba(229, 193, 88, 0.10), transparent 72%);
    animation: tfFogDriftOdd 16s ease-in-out infinite;
}
.tf-fog-2 {
    bottom: 9%;
    height: 200px;
    background: radial-gradient(ellipse 70% 65% at 68% 52%, rgba(0, 240, 255, 0.09), transparent 72%),
                radial-gradient(ellipse 55% 65% at 22% 55%, rgba(157, 78, 221, 0.09), transparent 70%);
    animation: tfFogDriftEven 21s ease-in-out infinite;
}
.tf-fog-3 {
    bottom: 20%;
    height: 230px;
    background: radial-gradient(ellipse 60% 55% at 48% 50%, rgba(229, 193, 88, 0.08), transparent 75%);
    animation: tfFogDriftOdd 26s ease-in-out infinite;
}
.tf-fog-4 {
    bottom: 34%;
    height: 240px;
    background: radial-gradient(ellipse 65% 50% at 35% 50%, rgba(0, 240, 255, 0.065), transparent 75%);
    animation: tfFogDriftEven 31s ease-in-out infinite;
}
.tf-fog-5 {
    bottom: 50%;
    height: 250px;
    background: radial-gradient(ellipse 60% 50% at 66% 45%, rgba(157, 78, 221, 0.065), transparent 76%);
    animation: tfFogDriftOdd 36s ease-in-out infinite;
}
.tf-fog-6 {
    bottom: 68%;
    height: 260px;
    background: radial-gradient(ellipse 70% 45% at 50% 40%, rgba(229, 193, 88, 0.055), transparent 78%);
    animation: tfFogDriftEven 42s ease-in-out infinite;
}

/* ==========================================================
   RIGHT-EDGE VERTICAL TECHFEST SCROLL SPINE (#tf-scroll-spine)
   ========================================================== */
.tf-scroll-spine {
    position: fixed;
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 995;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.65rem;
    padding: 0.85rem 0.4rem;
    background: rgba(2, 6, 15, 0.78);
    border: 1px solid rgba(229, 193, 88, 0.32);
    clip-path: polygon(0 8px, 8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%);
    backdrop-filter: blur(10px);
}

.tf-spine-node {
    width: 10px;
    height: 10px;
    display: block;
    background: rgba(148, 163, 184, 0.28);
    border: 1px solid rgba(0, 240, 255, 0.45);
    transform: rotate(45deg);
    transition: all 0.25s ease;
    cursor: pointer;
    position: relative;
}

.tf-spine-node:hover,
.tf-spine-node.active {
    background: #00f0ff;
    border-color: #fae084;
    box-shadow: 0 0 12px #00f0ff, 0 0 20px rgba(229, 193, 88, 0.65);
    transform: rotate(45deg) scale(1.35);
}

@media (max-width: 900px) {
    .tf-scroll-spine { display: none; }
}

/* ==========================================================
   TOP TECHFEST COMMAND HUD NAVBAR
   ========================================================== */
.tf-hud-navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 62px;
    background: linear-gradient(90deg, rgba(2, 5, 12, 0.92) 0%, rgba(8, 18, 36, 0.88) 50%, rgba(2, 5, 12, 0.92) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(229, 193, 88, 0.32);
    box-shadow: 0 6px 32px rgba(0, 0, 0, 0.7), 0 1px 0 rgba(0, 240, 255, 0.25);
    z-index: 998;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.2rem 0 4.2rem;
    pointer-events: auto;
}

.tf-hud-navbar::after {
    content: "";
    position: absolute;
    bottom: -2px;
    left: 0;
    height: 2px;
    width: calc(var(--tf-scroll-pct, 0) * 1%);
    background: linear-gradient(90deg, #00f0ff, #e5c158, #ff3b5c);
    box-shadow: 0 0 14px #00f0ff;
    transition: width 0.08s linear;
}

.tf-brand-cluster {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.tf-brand-emblem {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(229, 193, 88, 0.55);
    clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.35);
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}

.tf-brand-cluster:hover .tf-brand-emblem {
    transform: rotate(90deg) scale(1.12);
    box-shadow: 0 0 24px rgba(229, 193, 88, 0.75);
}

.tf-brand-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.96rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    color: #f8fafc;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.45);
}

.tf-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: var(--tf-gold);
    display: block;
}

.tf-nav-links {
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.tf-nav-link {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #cbd5e1 !important;
    text-decoration: none !important;
    padding: 0.42rem 0.82rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(8, 18, 36, 0.55);
    clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.tf-nav-link:hover {
    color: #02050b !important;
    background: linear-gradient(135deg, #00f0ff 0%, #e5c158 100%);
    border-color: #fae084;
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.55);
}

.tf-telemetry-cluster {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.tf-realm-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--tf-gold-bright);
    background: rgba(229, 193, 88, 0.1);
    border: 1px solid rgba(229, 193, 88, 0.45);
    padding: 0.32rem 0.78rem;
    clip-path: polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%);
    text-transform: uppercase;
}

.tf-scroll-readout {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--tf-cyan);
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.38);
    padding: 0.32rem 0.72rem;
}

@media (max-width: 1100px) {
    .tf-nav-links { display: none; }
}

/* ==========================================================
   TYPOGRAPHY
   ========================================================== */
h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: 0.04em;
}

p, li, label, .stCaption {
    font-family: 'Rajdhani', sans-serif !important;
    color: #94a3b8;
    font-size: 1.02rem;
}

/* ==========================================================
   HERO CITADEL STAGE (TECHFEST.ORG "AETHERIAL RENAISSANCE")
   ========================================================== */
.tf-hero-citadel {
    position: relative;
    padding: 3.2rem 2.6rem 2.4rem;
    margin-bottom: 1.5rem;
    background:
        radial-gradient(480px circle at var(--mx, 50%) var(--my, 50%), rgba(0, 240, 255, 0.14), transparent 60%),
        radial-gradient(ellipse 90% 75% at 50% 100%, rgba(0, 240, 255, 0.16), transparent 65%),
        radial-gradient(ellipse 65% 60% at 50% 20%, rgba(229, 193, 88, 0.14), transparent 70%),
        linear-gradient(180deg, rgba(4, 10, 24, 0.86) 0%, rgba(7, 19, 38, 0.92) 100%);
    border: 1px solid rgba(229, 193, 88, 0.42);
    clip-path: polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 28px 100%, 0 calc(100% - 28px));
    box-shadow:
        0 26px 70px rgba(0, 0, 0, 0.78),
        inset 0 0 60px rgba(0, 240, 255, 0.07);
    overflow: hidden;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.4s ease, box-shadow 0.4s ease;
}

.tf-hero-citadel:hover {
    border-color: rgba(0, 240, 255, 0.78);
    box-shadow:
        0 32px 90px rgba(0, 0, 0, 0.85),
        0 0 42px rgba(0, 240, 255, 0.24),
        inset 0 0 80px rgba(229, 193, 88, 0.11);
}

.tf-hero-grid {
    display: grid;
    grid-template-columns: 1.25fr 0.95fr;
    gap: 2.2rem;
    align-items: center;
    position: relative;
    z-index: 3;
}

@media (max-width: 960px) {
    .tf-hero-grid {
        grid-template-columns: 1fr;
    }
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--tf-gold-bright);
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    padding: 0.42rem 1rem;
    background: rgba(229, 193, 88, 0.1);
    border: 1px solid rgba(229, 193, 88, 0.45);
    clip-path: polygon(10px 0, 100% 0, calc(100% - 10px) 100%, 0 100%);
    margin-bottom: 1.15rem;
    box-shadow: 0 0 22px rgba(229, 193, 88, 0.18);
}

.hero-title {
    font-family: 'Cinzel', serif !important;
    color: #ffffff;
    font-size: 3.35rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #fae084 48%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 6px 24px rgba(0, 240, 255, 0.25));
}

.hero-description {
    max-width: 720px;
    color: #cbd5e1;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.14rem;
    font-weight: 500;
    line-height: 1.68;
    margin-bottom: 1.4rem;
}

.tf-hero-telemetry-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1.2rem;
}

.tf-telemetry-chip {
    background: rgba(3, 9, 20, 0.78);
    border: 1px solid rgba(0, 240, 255, 0.28);
    padding: 0.55rem 0.95rem;
    clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px);
    transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
}

.tf-telemetry-chip:hover {
    transform: translateY(-4px) scale(1.04);
    border-color: var(--tf-gold-bright);
    box-shadow: 0 10px 26px rgba(0, 240, 255, 0.28);
}

.tf-chip-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: var(--tf-muted);
    display: block;
}

.tf-chip-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.92rem;
    font-weight: 800;
    color: var(--tf-cyan);
}

.tf-hero-palace-art {
    position: relative;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    perspective: 1000px;
}

.tf-palace-emblem-svg {
    width: 100%;
    max-width: 390px;
    height: auto;
    filter: drop-shadow(0 0 28px rgba(0, 240, 255, 0.35));
    transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1), filter 0.5s ease;
}

.tf-hero-citadel:hover .tf-palace-emblem-svg {
    transform: rotateY(-8deg) rotateX(5deg) scale(1.05);
    filter: drop-shadow(0 0 42px rgba(229, 193, 88, 0.65));
}

.tf-scroll-prompt {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(229, 193, 88, 0.18);
    font-family: 'Orbitron', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.28em;
    color: var(--tf-gold);
    text-transform: uppercase;
}

.tf-scroll-chevron {
    display: inline-block;
    color: var(--tf-cyan);
    animation: tfScrollChevron 1.8s ease-in-out infinite;
}

/* ==========================================================
   SECTION HEADINGS & ARCHITECTURAL BANNERS
   ========================================================== */
.section-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--tf-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.24em !important;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    padding: 0.24rem 0.72rem;
    background: rgba(0, 240, 255, 0.08);
    border-left: 3px solid var(--tf-gold);
    border-right: 1px solid rgba(0, 240, 255, 0.25);
    transition: all 0.3s ease;
}

.section-kicker:hover {
    background: rgba(229, 193, 88, 0.16);
    color: var(--tf-gold-bright) !important;
    padding-left: 1rem;
    box-shadow: 0 0 18px rgba(0, 240, 255, 0.28);
}

.section-heading {
    font-family: 'Cinzel', serif !important;
    color: #f8fafc !important;
    font-size: 1.82rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 22px rgba(0, 240, 255, 0.22);
}

.section-description {
    color: #94a3b8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.02em;
    margin-bottom: 1.25rem;
}

hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, rgba(0, 240, 255, 0.45) 25%, rgba(229, 193, 88, 0.75) 50%, rgba(0, 240, 255, 0.45) 75%, transparent 100%) !important;
    margin: 2.2rem 0 !important;
    position: relative;
}

/* ==========================================================
   CHAMFERED CYBER-RENAISSANCE CARDS & HOVER EFFECTS
   ========================================================== */
.assessment-card,
.status-card,
[data-testid="stMetric"],
[data-testid="stFileUploader"] {
    position: relative !important;
    background:
        radial-gradient(380px circle at var(--mx, 50%) var(--my, 50%), rgba(0, 240, 255, 0.14), transparent 60%),
        linear-gradient(155deg, rgba(8, 19, 38, 0.92) 0%, rgba(3, 8, 18, 0.95) 100%) !important;
    border: 1px solid rgba(0, 240, 255, 0.26) !important;
    border-radius: 0px !important;
    clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px));
    padding: 1.45rem 1.5rem !important;
    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.65),
        inset 0 1px 0 rgba(255, 255, 255, 0.08),
        inset 0 -2px 0 rgba(229, 193, 88, 0.25) !important;
    transition:
        transform 0.32s cubic-bezier(0.16, 1, 0.3, 1),
        border-color 0.32s ease,
        box-shadow 0.32s ease !important;
    overflow: hidden !important;
}

.assessment-card::before,
.status-card::before,
[data-testid="stMetric"]::before {
    content: "◈";
    position: absolute;
    top: 8px;
    right: 14px;
    font-size: 0.65rem;
    color: rgba(229, 193, 88, 0.55);
    transition: transform 0.35s ease, color 0.35s ease;
    pointer-events: none;
}

.assessment-card::after,
.status-card::after,
[data-testid="stMetric"]::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: -20%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f0ff, #fae084, #00f0ff, transparent);
    box-shadow: 0 0 16px #00f0ff;
    opacity: 0;
    pointer-events: none;
}

.assessment-card:hover,
.status-card:hover,
[data-testid="stMetric"]:hover,
[data-testid="stFileUploader"]:hover {
    transform: perspective(900px) rotateX(var(--tilt-x, 2deg)) rotateY(var(--tilt-y, -2deg)) translateY(-6px) scale(1.015) !important;
    border-color: var(--tf-gold-bright) !important;
    box-shadow:
        0 24px 60px rgba(0, 0, 0, 0.85),
        0 0 32px rgba(0, 240, 255, 0.28),
        inset 0 0 28px rgba(229, 193, 88, 0.12) !important;
}

.assessment-card:hover::before,
.status-card:hover::before,
[data-testid="stMetric"]:hover::before {
    color: #00f0ff;
    transform: rotate(90deg) scale(1.35);
}

.assessment-card:hover::after,
.status-card:hover::after,
[data-testid="stMetric"]:hover::after {
    animation: tfLaserScan 1.6s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

.assessment-card {
    min-height: 182px;
}

.assessment-label,
.status-label,
[data-testid="stMetricLabel"] {
    color: var(--tf-gold) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}

.assessment-value,
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.05rem !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    margin-top: 0.65rem;
    text-shadow: 0 0 18px rgba(0, 240, 255, 0.35);
}

.assessment-subtitle {
    color: #cbd5e1 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.96rem !important;
    font-weight: 500;
    margin-top: 0.4rem;
}

/* ==========================================================
   STATUS CARDS (REFERABLE vs NON-REFERABLE)
   ========================================================== */
.status-card {
    min-height: 182px;
}

.status-card.referable {
    background:
        radial-gradient(360px circle at var(--mx, 50%) var(--my, 50%), rgba(255, 59, 92, 0.22), transparent 65%),
        linear-gradient(155deg, rgba(48, 8, 20, 0.92) 0%, rgba(18, 4, 10, 0.96) 100%) !important;
    border: 1px solid rgba(255, 59, 92, 0.55) !important;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.7), 0 0 24px rgba(255, 59, 92, 0.18) !important;
}

.status-card.normal {
    background:
        radial-gradient(360px circle at var(--mx, 50%) var(--my, 50%), rgba(0, 245, 155, 0.18), transparent 65%),
        linear-gradient(155deg, rgba(6, 38, 28, 0.92) 0%, rgba(3, 16, 12, 0.96) 100%) !important;
    border: 1px solid rgba(0, 245, 155, 0.48) !important;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.7), 0 0 24px rgba(0, 245, 155, 0.15) !important;
}

.status-value {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.72rem !important;
    font-weight: 800 !important;
    margin-top: 0.65rem;
    letter-spacing: 0.06em;
}

.status-card.referable .status-value {
    color: #ff6b81 !important;
    text-shadow: 0 0 20px rgba(255, 59, 92, 0.55);
}

.status-card.normal .status-value {
    color: #00f59b !important;
    text-shadow: 0 0 20px rgba(0, 245, 155, 0.55);
}

.status-description {
    color: #e2e8f0 !important;
    font-size: 0.94rem !important;
    margin-top: 0.45rem;
}

/* ==========================================================
   INPUTS, SELECTBOXES, RADIO & FILE UPLOADER
   ========================================================== */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background: rgba(5, 12, 26, 0.88) !important;
    border: 1px solid rgba(0, 240, 255, 0.25) !important;
    border-radius: 0px !important;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: all 0.25s ease !important;
}

[data-testid="stTextInput"] input:hover,
[data-testid="stNumberInput"] input:hover,
[data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--tf-gold-bright) !important;
    box-shadow: 0 0 18px rgba(0, 240, 255, 0.28) !important;
    background: rgba(10, 22, 46, 0.96) !important;
}

[data-testid="stWidgetLabel"] p {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase;
    color: #cbd5e1 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(4, 10, 22, 0.72) !important;
    border: 1px dashed rgba(229, 193, 88, 0.45) !important;
    border-radius: 0px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(0, 240, 255, 0.08) !important;
    border-color: #00f0ff !important;
    box-shadow: inset 0 0 28px rgba(0, 240, 255, 0.16);
}

/* ==========================================================
   TACTICAL CHAMFERED BUTTONS & DOWNLOAD CTA
   ========================================================== */
.stButton > button,
.stDownloadButton > button {
    position: relative !important;
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.16) 0%, rgba(229, 193, 88, 0.18) 100%) !important;
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: 1px solid var(--tf-gold-bright) !important;
    border-radius: 0px !important;
    padding: 0.85rem 1.6rem !important;
    clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 240, 255, 0.22) !important;
    transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #00f0ff 0%, #e5c158 100%) !important;
    color: #02050b !important;
    border-color: #ffffff !important;
    transform: translateY(-4px) scale(1.015) !important;
    box-shadow: 0 16px 40px rgba(0, 240, 255, 0.45), 0 0 30px rgba(229, 193, 88, 0.65) !important;
}

/* ==========================================================
   OPTICAL RETICLE IMAGE VIEWPORTS & TABS
   ========================================================== */
[data-testid="stImage"] {
    position: relative;
    border: 1px solid rgba(0, 240, 255, 0.32);
    border-radius: 0px !important;
    background: #02050b;
    padding: 6px;
    clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.75);
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

[data-testid="stImage"]:hover {
    border-color: var(--tf-gold-bright);
    transform: translateY(-5px) scale(1.015);
    box-shadow: 0 22px 55px rgba(0, 0, 0, 0.88), 0 0 32px rgba(0, 240, 255, 0.35);
}

[data-testid="stImage"] img {
    transition: transform 0.55s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stImage"]:hover img {
    transform: scale(1.035) !important;
}

button[data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #94a3b8 !important;
    padding: 0.65rem 1.2rem !important;
    transition: all 0.25s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--tf-gold-bright) !important;
    background: rgba(229, 193, 88, 0.08) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #00f0ff !important;
    border-bottom: 2px solid #00f0ff !important;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.55);
}

/* ==========================================================
   PROGRESS BARS & DATAFRAME / PLOTLY CONTAINERS
   ========================================================== */
.stProgress > div > div {
    background: rgba(15, 23, 42, 0.85) !important;
    border-radius: 0px !important;
    border: 1px solid rgba(0, 240, 255, 0.22);
    height: 10px !important;
    transition: transform 0.25s ease, border-color 0.25s ease;
}

.stProgress:hover > div > div {
    border-color: var(--tf-gold-bright);
    transform: scaleY(1.22);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00f0ff 0%, #e5c158 65%, #ff3b5c 100%) !important;
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.65);
}

[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"] {
    background: linear-gradient(155deg, rgba(6, 14, 29, 0.9) 0%, rgba(2, 6, 14, 0.94) 100%) !important;
    border: 1px solid rgba(0, 240, 255, 0.26) !important;
    clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px));
    padding: 0.75rem !important;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.7);
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stDataFrame"]:hover,
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-4px);
    border-color: var(--tf-gold-bright) !important;
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.85), 0 0 28px rgba(0, 240, 255, 0.22);
}

/* ==========================================================
   SIDEBAR CITADEL CONSOLE
   ========================================================== */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 50% 0%, rgba(0, 240, 255, 0.12), transparent 55%),
        linear-gradient(180deg, #030711 0%, #050d1d 55%, #02050b 100%) !important;
    border-right: 1px solid rgba(229, 193, 88, 0.32) !important;
    box-shadow: 8px 0 36px rgba(0, 0, 0, 0.8);
}

section[data-testid="stSidebar"] [data-testid="stMetric"] {
    margin-bottom: 0.85rem;
}

/* ==========================================================
   ALERTS & FOOTER
   ========================================================== */
[data-testid="stAlert"] {
    background: rgba(8, 19, 38, 0.88) !important;
    border: 1px solid rgba(0, 240, 255, 0.32) !important;
    border-left: 4px solid var(--tf-gold) !important;
    border-radius: 0px !important;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
    color: #e2e8f0 !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stAlert"]:hover {
    border-color: var(--tf-gold-bright) !important;
    box-shadow: 0 0 22px rgba(0, 240, 255, 0.2);
}

.app-footer {
    position: relative;
    text-align: center;
    color: #94a3b8;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    line-height: 2;
    padding: 2.6rem 1.5rem;
    margin-top: 3rem;
    background: linear-gradient(180deg, rgba(4, 10, 22, 0.78) 0%, rgba(2, 5, 11, 0.95) 100%);
    border-top: 1px solid rgba(229, 193, 88, 0.38);
    border-bottom: 1px solid rgba(0, 240, 255, 0.28);
    clip-path: polygon(18px 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%, 0 18px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.app-footer:hover {
    border-color: #00f0ff;
    box-shadow: 0 0 32px rgba(0, 240, 255, 0.2);
}
</style>

<!-- FIXED MULTI-LAYER TECHFEST PARALLAX STAGE (FARSIDE + PALACE BASE + PALACE GLOW + TRANSITION FOG 1-6) -->
<div class="tf-fixed-stage" aria-hidden="true">
    <div class="tf-layer-farside" id="tf-layer-farside"></div>
    <div class="tf-cyber-grid" id="tf-css-grid"></div>
    <div class="tf-farside-astrolabe" id="tf-css-astrolabe"></div>
    <div class="tf-fog-band tf-fog-6" id="tf-fog-6"></div>
    <div class="tf-fog-band tf-fog-5" id="tf-fog-5"></div>
    <div class="tf-fog-band tf-fog-4" id="tf-fog-4"></div>
    <div class="tf-fog-band tf-fog-3" id="tf-fog-3"></div>
    <div class="tf-fog-band tf-fog-2" id="tf-fog-2"></div>
    <div class="tf-palace-horizon" id="tf-css-palace">
        <svg viewBox="0 0 1920 360" width="100%" height="100%" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="palaceBaseGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#0b1d3a" stop-opacity="0.72"/>
                    <stop offset="100%" stop-color="#02050b" stop-opacity="0.98"/>
                </linearGradient>
                <linearGradient id="palaceGlowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.1"/>
                    <stop offset="50%" stop-color="#e5c158" stop-opacity="0.65"/>
                    <stop offset="100%" stop-color="#00f0ff" stop-opacity="0.1"/>
                </linearGradient>
            </defs>
            <!-- Palace Base Silhouette (Renaissance Citadel Spires & Arches) -->
            <path d="M0,360 L0,290 L90,290 L110,240 L130,290 L220,290 L240,210 L265,160 L290,210 L310,290 L430,290 L450,220 L490,180 L530,220 L550,290 L680,290 L710,195 L750,120 L775,70 L800,120 L840,195 L870,290 L910,290 L935,140 L960,35 L985,140 L1010,290 L1050,290 L1080,195 L1120,120 L1145,70 L1170,120 L1210,195 L1240,290 L1370,290 L1390,220 L1430,180 L1470,220 L1490,290 L1610,290 L1630,210 L1655,160 L1680,210 L1700,290 L1790,290 L1810,240 L1830,290 L1920,290 L1920,360 Z" fill="url(#palaceBaseGrad)" stroke="rgba(229,193,88,0.28)" stroke-width="1.2"/>
            <!-- Palace Glow Layer (Illuminated Ocular Rose Window & Spire Beacons) -->
            <g class="tf-palace-glow-svg">
                <circle cx="960" cy="185" r="44" fill="none" stroke="#00f0ff" stroke-width="1.6" stroke-dasharray="6 4"/>
                <circle cx="960" cy="185" r="24" fill="rgba(0,240,255,0.14)" stroke="#e5c158" stroke-width="1.5"/>
                <circle cx="960" cy="185" r="8" fill="#fae084"/>
                <circle cx="775" cy="70" r="4" fill="#00f0ff"/>
                <circle cx="960" cy="35" r="5" fill="#fae084"/>
                <circle cx="1145" cy="70" r="4" fill="#00f0ff"/>
                <line x1="200" y1="305" x2="1720" y2="305" stroke="url(#palaceGlowGrad)" stroke-width="2"/>
            </g>
        </svg>
    </div>
    <div class="tf-fog-band tf-fog-1" id="tf-fog-1"></div>
</div>

<!-- RIGHT-SIDE VERTICAL SCROLL TELEMETRY SPINE -->
<div class="tf-scroll-spine" aria-hidden="true">
    <a class="tf-spine-node active" href="#sec-hero" title="00 // Citadel Hero"></a>
    <a class="tf-spine-node" href="#sec-patient" title="01 // Patient Telemetry"></a>
    <a class="tf-spine-node" href="#sec-capture" title="02 // Optical Capture"></a>
    <a class="tf-spine-node" href="#sec-assessment" title="03 // AI Core"></a>
    <a class="tf-spine-node" href="#sec-validation" title="07 // Validation Benchmarks"></a>
    <a class="tf-spine-node" href="#sec-planning" title="08 // District Cartography"></a>
</div>

<!-- TOP FIXED TECHFEST HUD NAVIGATION BAR -->
<nav class="tf-hud-navbar">
    <div class="tf-brand-cluster">
        <div class="tf-brand-emblem">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M2 12C4.8 6.8 8.5 4 12 4C15.5 4 19.2 6.8 22 12C19.2 17.2 15.5 20 12 20C8.5 20 4.8 17.2 2 12Z" stroke="#00f0ff" stroke-width="1.8"/>
                <circle cx="12" cy="12" r="4" stroke="#e5c158" stroke-width="1.8"/>
                <circle cx="12" cy="12" r="1.5" fill="#ffffff"/>
            </svg>
        </div>
        <div>
            <span class="tf-brand-title">RETINASCOPE</span>
            <span class="tf-brand-sub">AETHERIAL RENAISSANCE // AI OCULAR CITADEL</span>
        </div>
    </div>
    <div class="tf-nav-links">
        <a class="tf-nav-link" href="#sec-hero">HOME</a>
        <a class="tf-nav-link" href="#sec-patient">01 // PATIENT</a>
        <a class="tf-nav-link" href="#sec-capture">02 // CAPTURE</a>
        <a class="tf-nav-link" href="#sec-assessment">03 // AI CORE</a>
        <a class="tf-nav-link" href="#sec-validation">07 // METRICS</a>
        <a class="tf-nav-link" href="#sec-planning">08 // CARTOGRAPHY</a>
    </div>
    <div class="tf-telemetry-cluster">
        <span class="tf-realm-pill" id="tf-realm-indicator">REALM I // FARSIDE HORIZON</span>
        <span class="tf-scroll-readout" id="tf-scroll-indicator">SCROLL // 00%</span>
    </div>
</nav>
""",
    unsafe_allow_html=True
)

# ============================================================
# INTERACTIVE SCROLL-REACTIVE PARALLAX & 3D HOVER JS ENGINE
# ============================================================
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        const win = window.parent;
        if (!doc || doc.getElementById('tf-aether-canvas')) return;

        // 1. Inject Full-Viewport Interactive Aether Canvas
        const canvas = doc.createElement('canvas');
        canvas.id = 'tf-aether-canvas';
        Object.assign(canvas.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: '1',
            opacity: '0.88'
        });
        doc.body.appendChild(canvas);
        const ctx = canvas.getContext('2d');

        let width = canvas.width = win.innerWidth;
        let height = canvas.height = win.innerHeight;
        win.addEventListener('resize', () => {
            width = canvas.width = win.innerWidth;
            height = canvas.height = win.innerHeight;
        }, { passive: true });

        // 5 Scroll-Driven Realms (Background Change by Scroll)
        const REALMS = [
            { stop: 0.00, name: "REALM I // FARSIDE HORIZON", top: [2, 5, 13], mid: [6, 21, 45], bot: [11, 34, 57], nebA: "rgba(0, 240, 255, 0.17)", nebB: "rgba(229, 193, 88, 0.13)" },
            { stop: 0.25, name: "REALM II // PALACE SANCTUARY", top: [3, 12, 24], mid: [7, 38, 58], bot: [10, 54, 70], nebA: "rgba(0, 255, 200, 0.17)", nebB: "rgba(0, 240, 255, 0.14)" },
            { stop: 0.50, name: "REALM III // OCULAR CRUCIBLE", top: [12, 4, 26], mid: [30, 11, 54], bot: [46, 14, 68], nebA: "rgba(192, 132, 252, 0.18)", nebB: "rgba(255, 59, 92, 0.14)" },
            { stop: 0.75, name: "REALM IV // CITADEL ARCHIVES", top: [2, 14, 17], mid: [6, 40, 45], bot: [10, 60, 64], nebA: "rgba(0, 245, 155, 0.16)", nebB: "rgba(0, 240, 255, 0.14)" },
            { stop: 1.00, name: "REALM V // TERRA CARTOGRAPHY", top: [13, 8, 3], mid: [36, 23, 8], bot: [54, 34, 11], nebA: "rgba(229, 193, 88, 0.20)", nebB: "rgba(255, 140, 0, 0.15)" }
        ];

        function lerp(a, b, t) { return a + (b - a) * t; }
        function lerpRGB(c1, c2, t) {
            return `rgb(${Math.round(lerp(c1[0], c2[0], t))}, ${Math.round(lerp(c1[1], c2[1], t))}, ${Math.round(lerp(c1[2], c2[2], t))})`;
        }

        let scrollPct = 0;
        let scrollPx = 0;
        let lastScrollPx = 0;
        let scrollVel = 0;
        let mouseX = width * 0.5;
        let mouseY = height * 0.35;

        const FOG_SPEEDS = [-0.08, -0.14, -0.21, -0.29, -0.38, -0.48];

        function computeScrollMetrics() {
            const containers = [
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('.main'),
                doc.documentElement,
                doc.body
            ];
            let maxScroll = 1;
            let currentTop = 0;
            for (const el of containers) {
                if (!el) continue;
                const sh = el.scrollHeight - el.clientHeight;
                if (sh > maxScroll) maxScroll = sh;
                if (el.scrollTop > currentTop) currentTop = el.scrollTop;
            }
            if (win.scrollY > currentTop) currentTop = win.scrollY;
            scrollPx = currentTop;
            scrollPct = Math.max(0, Math.min(100, (currentTop / Math.max(1, maxScroll)) * 100));
            scrollVel = Math.max(-45, Math.min(45, scrollPx - lastScrollPx));
            lastScrollPx = scrollPx;

            const norm = scrollPct / 100;
            let r0 = REALMS[0], r1 = REALMS[REALMS.length - 1];
            for (let i = 0; i < REALMS.length - 1; i++) {
                if (norm >= REALMS[i].stop && norm <= REALMS[i + 1].stop) {
                    r0 = REALMS[i];
                    r1 = REALMS[i + 1];
                    break;
                }
            }
            const localT = (norm - r0.stop) / Math.max(0.001, (r1.stop - r0.stop));
            const chosen = localT < 0.5 ? r0 : r1;
            const root = doc.documentElement;
            root.style.setProperty('--tf-scroll-pct', scrollPct.toFixed(2));
            root.style.setProperty('--tf-scroll-px', `${scrollPx.toFixed(1)}px`);
            root.style.setProperty('--tf-sky-top', lerpRGB(r0.top, r1.top, localT));
            root.style.setProperty('--tf-sky-mid', lerpRGB(r0.mid, r1.mid, localT));
            root.style.setProperty('--tf-sky-bot', lerpRGB(r0.bot, r1.bot, localT));
            root.style.setProperty('--tf-nebula-a', chosen.nebA);
            root.style.setProperty('--tf-nebula-b', chosen.nebB);

            const realmEl = doc.getElementById('tf-realm-indicator');
            if (realmEl) realmEl.textContent = chosen.name;
            const scrollEl = doc.getElementById('tf-scroll-indicator');
            if (scrollEl) scrollEl.textContent = `SCROLL // ${String(Math.round(scrollPct)).padStart(2, '0')}%`;

            // Update Right-Edge Scroll Spine active node
            const spineNodes = doc.querySelectorAll('.tf-spine-node');
            if (spineNodes.length > 0) {
                const activeNodeIdx = Math.min(spineNodes.length - 1, Math.floor(norm * spineNodes.length));
                spineNodes.forEach((n, idx) => n.classList.toggle('active', idx === activeNodeIdx));
            }

            // Parallax transforms on Farside Astrolabe, Palace Base/Glow & 6 Transition Fog Layers
            const astrolabe = doc.getElementById('tf-css-astrolabe');
            if (astrolabe) {
                astrolabe.style.transform = `translate(-50%, calc(-50% + ${scrollPx * -0.14}px)) rotate(${scrollPx * 0.08}deg)`;
            }
            const palace = doc.getElementById('tf-css-palace');
            if (palace) {
                const scale = 1 + (scrollPct * 0.0014);
                palace.style.transform = `translate3d(0, ${Math.min(65, scrollPx * 0.04)}px, 0) scale(${scale.toFixed(3)})`;
            }
            for (let f = 1; f <= 6; f++) {
                const fogEl = doc.getElementById(`tf-fog-${f}`);
                if (fogEl) {
                    const yShift = (scrollPx * FOG_SPEEDS[f - 1]) % 260;
                    const xShift = Math.sin((scrollPx * 0.004) + f) * 28;
                    fogEl.style.transform = `translate3d(${xShift.toFixed(1)}px, ${yShift.toFixed(1)}px, 0)`;
                }
            }
        }

        doc.addEventListener('scroll', computeScrollMetrics, { capture: true, passive: true });
        win.addEventListener('scroll', computeScrollMetrics, { passive: true });

        // 2. 3D Card Tilt & Radial Cursor Spotlight on Hover
        doc.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
            const target = e.target ? e.target.closest('.assessment-card, .status-card, [data-testid="stMetric"], [data-testid="stFileUploader"], .tf-hero-citadel') : null;
            if (target) {
                const rect = target.getBoundingClientRect();
                const relX = e.clientX - rect.left;
                const relY = e.clientY - rect.top;
                const pctX = (relX / Math.max(1, rect.width)) - 0.5;
                const pctY = (relY / Math.max(1, rect.height)) - 0.5;
                target.style.setProperty('--mx', `${relX}px`);
                target.style.setProperty('--my', `${relY}px`);
                target.style.setProperty('--tilt-x', `${(-pctY * 7).toFixed(2)}deg`);
                target.style.setProperty('--tilt-y', `${(pctX * 7).toFixed(2)}deg`);
            }
        }, { passive: true });

        // 3. 60fps Particle, Fog Plume & Scroll-Warp Canvas Loop
        const PARTICLES = Array.from({ length: 82 }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            r: Math.random() * 2.1 + 0.6,
            vx: (Math.random() - 0.5) * 0.35,
            vy: -Math.random() * 0.45 - 0.15,
            gold: Math.random() > 0.48,
            alpha: Math.random() * 0.55 + 0.2
        }));

        function renderLoop() {
            ctx.clearRect(0, 0, width, height);
            scrollVel *= 0.9;

            const cursorGrad = ctx.createRadialGradient(mouseX, mouseY, 2, mouseX, mouseY, 220);
            cursorGrad.addColorStop(0, 'rgba(0, 240, 255, 0.09)');
            cursorGrad.addColorStop(0.5, 'rgba(229, 193, 88, 0.04)');
            cursorGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = cursorGrad;
            ctx.fillRect(0, 0, width, height);

            for (let i = 0; i < PARTICLES.length; i++) {
                const p = PARTICLES[i];
                p.x += p.vx;
                p.y += p.vy - scrollVel * 0.38;

                if (p.x < 0) p.x = width;
                if (p.x > width) p.x = 0;
                if (p.y < 0) p.y = height;
                if (p.y > height) p.y = 0;

                const warpLen = Math.min(28, Math.abs(scrollVel) * 1.35);
                ctx.beginPath();
                ctx.strokeStyle = p.gold
                    ? `rgba(229, 193, 88, ${p.alpha})`
                    : `rgba(0, 240, 255, ${p.alpha})`;
                ctx.lineWidth = p.r;
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p.x, p.y + (scrollVel >= 0 ? warpLen : -warpLen) + 0.5);
                ctx.stroke();

                const dx = p.x - mouseX;
                const dy = p.y - mouseY;
                const dist = Math.hypot(dx, dy);
                if (dist < 145) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(0, 240, 255, ${(1 - dist / 145) * 0.25})`;
                    ctx.lineWidth = 0.7;
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(mouseX, mouseY);
                    ctx.stroke();
                }
            }
            win.requestAnimationFrame(renderLoop);
        }

        computeScrollMetrics();
        win.requestAnimationFrame(renderLoop);
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()



# ============================================================
# PATIENT SCREENING REPORT
# ============================================================

def generate_screening_report(patient, original_image, heatmap, probabilities,
                              predicted_class, confidence):
    """Create a professional PDF screening report in memory."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="RetinaScope Patient Screening Report",
        author="RetinaScope"
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20,
        leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"),
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=9,
        leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"),
        spaceAfter=10
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=11,
        leading=14, textColor=colors.HexColor("#2563EB"),
        spaceBefore=7, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=8.5,
        leading=12, textColor=colors.HexColor("#334155")
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=7,
        leading=9, textColor=colors.HexColor("#64748B")
    )

    story = [
        Paragraph("RETINASCOPE", title_style),
        Paragraph("Diabetic Retinopathy Screening Report", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5)
    ]

    patient_rows = [
        ["Case ID", patient["Case ID"], "Screening Date", patient["Screening Date"]],
        ["Patient Name", patient["Patient Name"], "Age", patient["Age"]],
        ["Sex", patient["Sex"], "Screening Centre", patient["Screening Centre"]],
        ["Eye Examined", patient["Eye Examined"], "Image ID", patient["Image ID"]],
        ["Diabetes Type", patient["Diabetes Type"], "Duration", patient["Diabetes Duration"]],
        ["Previous DR", patient["Previous DR"], "Previous Eye Exam", patient["Previous Eye Exam"]],
    ]

    story += [
        Paragraph("1. Patient Information", section_style),
        Table(patient_rows, colWidths=[30*mm, 57*mm, 34*mm, 57*mm],
              style=TableStyle([
                  ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
                  ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F5F9")),
                  ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
                  ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                  ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                  ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                  ("FONTSIZE",(0,0),(-1,-1),8),
                  ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
                  ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                  ("TOPPADDING",(0,0),(-1,-1),5),
                  ("BOTTOMPADDING",(0,0),(-1,-1),5),
              ]))
    ]

    referable = predicted_class >= 2
    status = "REFERABLE DR" if referable else "NON-REFERABLE"
    status_color = "#DC2626" if referable else "#16A34A"
    recommendation = (
        "Further evaluation by a qualified eye-care professional is recommended."
        if referable else
        "Routine screening / follow-up according to appropriate clinical guidance."
    )

    story += [
        Paragraph("2. AI Screening Result", section_style),
        Table([
            ["Predicted Grade", f"Grade {predicted_class} — {CLASS_NAMES[predicted_class]}"],
            ["Model Confidence", f"{confidence * 100:.2f}%"],
            ["Screening Status", status],
            ["Screening Recommendation", recommendation],
        ], colWidths=[48*mm, 130*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(1,0),(1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
            ("TEXTCOLOR",(1,2),(1,2),colors.HexColor(status_color)),
            ("FONTNAME",(1,2),(1,2),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ]))
    ]

    # Save image bytes for ReportLab
    def image_bytes(img):
        b = BytesIO()
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(b, format="PNG")
        b.seek(0)
        return b

    orig_bytes = BytesIO()
    original_image.save(orig_bytes, format="PNG")
    orig_bytes.seek(0)

    heat_bytes = image_bytes(heatmap)

    story += [
        Paragraph("3. Explainability", section_style),
        Table([
            [RLImage(orig_bytes, width=82*mm, height=62*mm),
             RLImage(heat_bytes, width=82*mm, height=62*mm)],
            [Paragraph("<b>Original Fundus Image</b>", small_style),
             Paragraph("<b>Grad-CAM Attention</b>", small_style)]
        ], colWidths=[89*mm, 89*mm], style=TableStyle([
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("BOX",(0,0),(-1,-2),0.5,colors.HexColor("#CBD5E1")),
            ("INNERGRID",(0,0),(-1,-2),0.4,colors.HexColor("#E2E8F0")),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph(
            "Grad-CAM highlights image regions that contributed to the selected model prediction. "
            "It represents model attention and is not a confirmed lesion map.",
            small_style
        )
    ]

    prob_rows = [["Severity", "Probability"]]
    for i in range(5):
        prob_rows.append([f"Grade {i} — {CLASS_NAMES[i]}", f"{float(probabilities[0, i]) * 100:.2f}%"])

    story += [
        Paragraph("4. Class Probability Distribution", section_style),
        Table(prob_rows, colWidths=[125*mm, 53*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("ALIGN",(1,1),(1,-1),"RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
    ]

    story += [
        Paragraph("5. Model Information", section_style),
        Paragraph(
            "Model: EfficientNet-B0 &nbsp;|&nbsp; Task: 5-class DR classification "
            "&nbsp;|&nbsp; Training dataset: APTOS 2019 &nbsp;|&nbsp; Explainability: Grad-CAM",
            body_style
        ),
        Spacer(1, 4),
        Paragraph(
            "Validation reference: 80.22% 5-class accuracy, 65.03% macro F1, "
            "90.27% referable sensitivity, 92.18% referable specificity.",
            body_style
        ),
        Spacer(1, 7),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5),
        Paragraph(
            "<b>IMPORTANT:</b> RetinaScope is a research and demonstration prototype. "
            "This AI-assisted screening output is not a standalone clinical diagnosis and "
            "should not replace evaluation by a qualified healthcare professional. "
            "Reported performance is based on an internal APTOS validation split and does "
            "not establish clinical validity.",
            small_style
        ),
        Spacer(1, 5),
        Paragraph("Generated by RetinaScope · Explainable AI for Diabetic Retinopathy Screening", small_style)
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()



# ============================================================
# DUAL-EYE PATIENT SCREENING REPORT
# ============================================================

def generate_dual_eye_report(patient, right_data, left_data):
    """Create a professional PDF report for bilateral screening."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="RetinaScope Dual-Eye Screening Report",
        author="RetinaScope"
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DualTitle", parent=styles["Title"], fontSize=20, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "DualSubtitle", parent=styles["Normal"], fontSize=9, leading=12,
        alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=10
    )
    section_style = ParagraphStyle(
        "DualSection", parent=styles["Heading2"], fontSize=11, leading=14,
        textColor=colors.HexColor("#2563EB"), spaceBefore=7, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "DualBody", parent=styles["Normal"], fontSize=8.5, leading=12,
        textColor=colors.HexColor("#334155")
    )
    small_style = ParagraphStyle(
        "DualSmall", parent=styles["Normal"], fontSize=7, leading=9,
        textColor=colors.HexColor("#64748B")
    )

    story = [
        Paragraph("RETINASCOPE", title_style),
        Paragraph("Dual-Eye Diabetic Retinopathy Screening Report", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5),
    ]

    patient_rows = [
        ["Case ID", patient["Case ID"], "Screening Date", patient["Screening Date"]],
        ["Patient Name", patient["Patient Name"], "Age", patient["Age"]],
        ["Sex", patient["Sex"], "Screening Centre", patient["Screening Centre"]],
        ["Eye Examined", "Both", "Right Image ID", patient["Right Image ID"]],
        ["Diabetes Type", patient["Diabetes Type"], "Left Image ID", patient["Left Image ID"]],
        ["Duration", patient["Diabetes Duration"], "Previous DR", patient["Previous DR"]],
        ["Previous Eye Exam", patient["Previous Eye Exam"], "", ""],
    ]
    story += [
        Paragraph("1. Patient Information", section_style),
        Table(patient_rows, colWidths=[30*mm,57*mm,34*mm,57*mm],
              style=TableStyle([
                  ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
                  ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F5F9")),
                  ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                  ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                  ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                  ("FONTSIZE",(0,0),(-1,-1),8),
                  ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
                  ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
                  ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                  ("TOPPADDING",(0,0),(-1,-1),5),
                  ("BOTTOMPADDING",(0,0),(-1,-1),5),
              ]))
    ]

    def result(d):
        pred = d["predicted_class"]
        return (
            f"Grade {pred} — {CLASS_NAMES[pred]}",
            f"{d['confidence']*100:.2f}%",
            "REFERABLE DR" if pred >= 2 else "NON-REFERABLE"
        )

    rr = result(right_data)
    lr = result(left_data)
    story += [
        Paragraph("2. AI Screening Result — Right vs Left Eye", section_style),
        Table([
            ["", "Right Eye", "Left Eye"],
            ["Predicted Grade", rr[0], lr[0]],
            ["Model Confidence", rr[1], lr[1]],
            ["Screening Status", rr[2], lr[2]],
        ], colWidths=[55*mm,62*mm,62*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ])),
        Spacer(1,5),
        Paragraph(
            "Inter-eye severity comparison describes the two AI screening outputs. "
            "It is not a clinical assessment of disease asymmetry.", small_style
        )
    ]

    def img_bytes(img):
        b = BytesIO()
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(b, format="PNG")
        b.seek(0)
        return b

    ro = BytesIO(); right_data["image"].save(ro, format="PNG"); ro.seek(0)
    lo = BytesIO(); left_data["image"].save(lo, format="PNG"); lo.seek(0)
    rh = img_bytes(right_data["heatmap"])
    lh = img_bytes(left_data["heatmap"])

    story += [
        Paragraph("3. Dual-Eye Explainability", section_style),
        Table([
            [Paragraph("<b>Right Eye</b>", small_style), Paragraph("<b>Left Eye</b>", small_style)],
            [RLImage(ro, width=76*mm, height=57*mm), RLImage(lo, width=76*mm, height=57*mm)],
            [RLImage(rh, width=76*mm, height=57*mm), RLImage(lh, width=76*mm, height=57*mm)],
            [Paragraph("Original Fundus", small_style), Paragraph("Original Fundus", small_style)],
            [Paragraph("Grad-CAM Attention", small_style), Paragraph("Grad-CAM Attention", small_style)],
        ], colWidths=[89*mm,89*mm], style=TableStyle([
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
            ("INNERGRID",(0,0),(-1,-1),0.4,colors.HexColor("#E2E8F0")),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph(
            "Grad-CAM highlights regions that contributed to each selected prediction. "
            "Highlighted regions represent model attention, not confirmed lesions.", small_style
        )
    ]

    prob_rows = [["Severity", "Right Eye", "Left Eye"]]
    for i in range(5):
        prob_rows.append([
            f"Grade {i} — {CLASS_NAMES[i]}",
            f"{float(right_data['probabilities'][0,i])*100:.2f}%",
            f"{float(left_data['probabilities'][0,i])*100:.2f}%"
        ])

    story += [
        Paragraph("4. Class Probability Comparison", section_style),
        Table(prob_rows, colWidths=[95*mm,41*mm,41*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("ALIGN",(1,1),(-1,-1),"RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph("5. Model Information", section_style),
        Paragraph(
            "Model: EfficientNet-B0 &nbsp;|&nbsp; Task: 5-class DR classification "
            "&nbsp;|&nbsp; Training dataset: APTOS 2019 &nbsp;|&nbsp; Explainability: Grad-CAM",
            body_style
        ),
        Spacer(1,4),
        Paragraph(
            "Validation reference: 80.22% 5-class accuracy, 65.03% macro F1, "
            "90.27% referable sensitivity, 92.18% referable specificity.", body_style
        ),
        Spacer(1,7),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#CBD5E1")),
        Spacer(1,5),
        Paragraph(
            "<b>IMPORTANT:</b> RetinaScope is a research and demonstration prototype. "
            "This AI-assisted screening output is not a standalone clinical diagnosis and "
            "should not replace evaluation by a qualified healthcare professional. "
            "Reported performance is based on an internal APTOS validation split and does "
            "not establish clinical validity.", small_style
        ),
        Spacer(1,5),
        Paragraph("Generated by RetinaScope · Explainable AI for Diabetic Retinopathy Screening", small_style)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = np.array(
        image.convert("RGB")
    )

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # CLAHE
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # Light denoising
    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Visualization image
    rgb_image = image.astype(
        np.float32
    ) / 255.0

    # Tensor
    tensor = torch.from_numpy(
        rgb_image
    ).permute(2, 0, 1).float()

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    tensor = tensor.unsqueeze(0)

    return rgb_image, tensor


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    input_tensor,
    predicted_class,
    rgb_image
):

    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization



# ============================================================
# DISTRICT RESOURCE PLANNING
# ============================================================

PLANNING_PATH = "planning_data.csv"
SHAPEFILE_PATH = "district_shapes/output.shp"

@st.cache_data
def load_planning_data():
    return pd.read_csv(PLANNING_PATH)

@st.cache_data
def load_district_shapes():
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def normalize_name(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace("&", "AND", regex=False)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

@st.cache_data(show_spinner=False)
def build_map_data(selected_state="All States"):
    planning = load_planning_data().copy()
    shapes = load_district_shapes().copy()

    # Normalize names for a robust state + district join.
    shapes["_state_key"] = normalize_name(shapes["statename"])
    shapes["_district_key"] = normalize_name(shapes["distname"])

    planning["_state_key"] = normalize_name(planning["state"])
    planning["_district_key"] = normalize_name(planning["district"])

    # IMPORTANT: filter before the merge so the browser receives only the
    # selected state's geometries instead of the complete India shapefile.
    if selected_state != "All States":
        state_key = normalize_name(pd.Series([selected_state])).iloc[0]
        shapes = shapes[shapes["_state_key"] == state_key].copy()
        planning = planning[planning["_state_key"] == state_key].copy()

    merged = shapes.merge(
        planning,
        on=["_state_key", "_district_key"],
        how="inner",
        suffixes=("_shape", "_plan")
    )

    merged = merged.dropna(subset=["priority"]).copy()

    if merged.empty:
        return merged

    # Reduce geometry complexity for much faster Plotly rendering.
    merged["geometry"] = merged.geometry.simplify(
        tolerance=0.005,
        preserve_topology=True
    )

    merged["map_id"] = merged.index.astype(str)
    return merged





# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 0.4rem 0 0.6rem;">
            <div style="font-family:'Orbitron',sans-serif;font-size:0.65rem;letter-spacing:0.24em;color:#e5c158;">
                CITADEL TELEMETRY // v30.0
            </div>
            <div style="font-family:'Cinzel',serif;font-size:1.55rem;font-weight:800;color:#ffffff;letter-spacing:0.06em;margin-top:0.2rem;">
                RETINASCOPE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Explainable AI for Diabetic Retinopathy Screening"
    )

    st.divider()

    st.markdown("### ◈ MODEL CORE")

    st.write("EfficientNet-B0")

    st.caption(
        "5-class severity classification"
    )

    st.divider()

    st.markdown("### ◈ VALIDATION PERFORMANCE")

    st.metric(
        "5-Class Accuracy",
        "80.22%"
    )

    st.metric(
        "Referable Sensitivity",
        "90.27%"
    )

    st.metric(
        "Referable Specificity",
        "92.18%"
    )

    st.divider()

    st.markdown("### ◈ SCREENING LOGIC")

    st.caption(
        "Grades 0–1 → Non-Referable"
    )

    st.caption(
        "Grades 2–4 → Referable"
    )

    st.divider()

    st.caption(
        "Research prototype. Not a clinical diagnostic device."
    )


# ============================================================
# HERO (TECHFEST "AETHERIAL RENAISSANCE" CITADEL STAGE)
# ============================================================

st.markdown(
    """
<div id="sec-hero" class="tf-hero-citadel">
    <div class="tf-hero-grid">
        <div>
            <div class="hero-eyebrow">
                ◈ RETINASCOPE · AN AETHERIAL RENAISSANCE OF AI SCREENING ◈
            </div>
            <div class="hero-title">
                Explainable Diabetic<br>
                Retinopathy Screening
            </div>
            <div class="hero-description">
                AI-assisted retinal image analysis with five-level severity
                classification, confidence estimation, and visual Grad-CAM
                explanations. Engineered as an interactive clinical-research observatory.
            </div>
            <div class="tf-hero-telemetry-strip">
                <div class="tf-telemetry-chip">
                    <span class="tf-chip-kicker">ARCHITECTURE</span>
                    <span class="tf-chip-val">EFFICIENTNET-B0</span>
                </div>
                <div class="tf-telemetry-chip">
                    <span class="tf-chip-kicker">SENSITIVITY</span>
                    <span class="tf-chip-val">90.27%</span>
                </div>
                <div class="tf-telemetry-chip">
                    <span class="tf-chip-kicker">SPECIFICITY</span>
                    <span class="tf-chip-val">92.18%</span>
                </div>
                <div class="tf-telemetry-chip">
                    <span class="tf-chip-kicker">EXPLAINABILITY</span>
                    <span class="tf-chip-val">GRAD-CAM // CONV_HEAD</span>
                </div>
            </div>
        </div>
        <div class="tf-hero-palace-art">
            <svg class="tf-palace-emblem-svg" viewBox="0 0 420 340" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="ocularCore" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.65"/>
                        <stop offset="55%" stop-color="#e5c158" stop-opacity="0.25"/>
                        <stop offset="100%" stop-color="#02050b" stop-opacity="0"/>
                    </radialGradient>
                </defs>
                <!-- Outer Sacred Astrolabe Rings -->
                <circle cx="210" cy="165" r="138" stroke="rgba(229,193,88,0.35)" stroke-width="1.2" stroke-dasharray="8 6"/>
                <circle cx="210" cy="165" r="112" stroke="rgba(0,240,255,0.45)" stroke-width="1.5"/>
                <circle cx="210" cy="165" r="88" fill="url(#ocularCore)" stroke="rgba(229,193,88,0.65)" stroke-width="1.2" stroke-dasharray="4 4"/>
                <!-- Cyber-Renaissance Citadel Spire Frame -->
                <polygon points="210,14 238,68 182,68" fill="rgba(229,193,88,0.18)" stroke="#e5c158" stroke-width="1.4"/>
                <polygon points="210,316 238,262 182,262" fill="rgba(0,240,255,0.15)" stroke="#00f0ff" stroke-width="1.4"/>
                <!-- Ocular Retinal Aperture -->
                <path d="M92 165C126 112 166 88 210 88C254 88 294 112 328 165C294 218 254 242 210 242C166 242 126 218 92 165Z" fill="rgba(4,12,26,0.75)" stroke="#00f0ff" stroke-width="2.2"/>
                <circle cx="210" cy="165" r="46" stroke="#fae084" stroke-width="2"/>
                <circle cx="210" cy="165" r="22" fill="rgba(0,240,255,0.28)" stroke="#00f0ff" stroke-width="1.8"/>
                <circle cx="210" cy="165" r="7" fill="#ffffff"/>
                <!-- Coordinate Reticle Ticks -->
                <line x1="40" y1="165" x2="82" y2="165" stroke="#e5c158" stroke-width="1.5"/>
                <line x1="338" y1="165" x2="380" y2="165" stroke="#e5c158" stroke-width="1.5"/>
                <text x="210" y="334" text-anchor="middle" fill="#e5c158" font-family="Orbitron, sans-serif" font-size="9" letter-spacing="3">PALACE OCULAR OBSERVATORY // 224×224 LAB-CLAHE</text>
            </svg>
        </div>
    </div>
    <div class="tf-scroll-prompt">
        <span>SCROLL TO DESCEND THROUGH THE SCREENING REALMS</span>
        <span class="tf-scroll-chevron">▼</span>
    </div>
</div>
""",
    unsafe_allow_html=True
)

st.divider()



# ============================================================
# PATIENT DETAILS
# ============================================================

st.markdown(
    '<div id="sec-patient" class="section-kicker">01 · PATIENT TELEMETRY</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-heading">Patient & Screening Details</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Enter the screening details that will appear on the generated patient report.</div>',
    unsafe_allow_html=True
)

case_id = st.text_input(
    "Case ID",
    value=f"RS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    help="Automatically generated. You can replace it with your institution's case ID."
)

p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
with p2:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
with p3:
    patient_sex = st.selectbox("Sex", ["Female", "Male", "Other", "Prefer not to say"])

p4, p5, p6 = st.columns(3, gap="medium")
with p4:
    screening_centre = st.text_input("Screening Centre", placeholder="Centre / Hospital name")
with p5:
    eye_examined = st.selectbox("Eye Examined", ["Right", "Left", "Both", "Unknown"])
with p6:
    diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2", "Other", "Unknown"])

p7, p8, p9 = st.columns(3, gap="medium")
with p7:
    diabetes_duration = st.text_input("Diabetes Duration", placeholder="e.g. 8 years")
with p8:
    previous_dr = st.selectbox("Previous DR History", ["Yes", "No", "Unknown"])
with p9:
    previous_eye_exam = st.selectbox("Previous Eye Examination", ["Yes", "No", "Unknown"])

st.divider()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div id="sec-capture" class="section-kicker">02 · OPTICAL INPUT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">Upload Fundus Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Upload a retinal fundus photograph in JPG, JPEG, or PNG format.</div>',
    unsafe_allow_html=True
)

screening_mode = st.radio(
    "Screening Mode",
    ["Single Eye", "Both Eyes"],
    horizontal=True,
    help="Analyse one eye or upload separate right- and left-eye fundus images."
)

if screening_mode == "Single Eye":
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png"],
        key="single_eye_upload"
    )
    right_eye_file = None
    left_eye_file = None
else:
    up_right, up_left = st.columns(2, gap="medium")
    with up_right:
        right_eye_file = st.file_uploader(
            "Right Eye — Fundus Image",
            type=["jpg", "jpeg", "png"],
            key="right_eye_upload"
        )
    with up_left:
        left_eye_file = st.file_uploader(
            "Left Eye — Fundus Image",
            type=["jpg", "jpeg", "png"],
            key="left_eye_upload"
        )
    uploaded_file = None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def run_eye_analysis(uploaded, generate_cam=True):
    """Run RetinaScope inference for one eye, with optional Grad-CAM."""
    img = Image.open(uploaded).convert("RGB")
    rgb, tensor = preprocess_image(img)

    with torch.inference_mode():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    cam = generate_gradcam(tensor, pred, rgb) if generate_cam else None

    return {
        "image": img,
        "rgb_image": rgb,
        "input_tensor": tensor,
        "probabilities": probs,
        "predicted_class": pred,
        "confidence": conf,
        "heatmap": cam,
        "image_id": uploaded.name,
    }


def run_dual_eye_analysis(right_uploaded, left_uploaded):
    """Batch EfficientNet inference for both eyes, then generate Grad-CAM."""
    right_img = Image.open(right_uploaded).convert("RGB")
    left_img = Image.open(left_uploaded).convert("RGB")

    right_rgb, right_tensor = preprocess_image(right_img)
    left_rgb, left_tensor = preprocess_image(left_img)

    # One EfficientNet forward pass for both eyes.
    batch = torch.cat([right_tensor, left_tensor], dim=0)

    with torch.inference_mode():
        output = model(batch)
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    right_pred = int(predictions[0].item())
    left_pred = int(predictions[1].item())
    right_conf = float(probabilities[0, right_pred].item())
    left_conf = float(probabilities[1, left_pred].item())

    # Grad-CAM is still the expensive CPU operation, but runs only after
    # the fast shared inference pass has completed.
    right_heatmap = generate_gradcam(
        right_tensor, right_pred, right_rgb
    )
    left_heatmap = generate_gradcam(
        left_tensor, left_pred, left_rgb
    )

    right_data = {
        "image": right_img,
        "rgb_image": right_rgb,
        "input_tensor": right_tensor,
        "probabilities": probabilities[0:1],
        "predicted_class": right_pred,
        "confidence": right_conf,
        "heatmap": right_heatmap,
        "image_id": right_uploaded.name,
    }

    left_data = {
        "image": left_img,
        "rgb_image": left_rgb,
        "input_tensor": left_tensor,
        "probabilities": probabilities[1:2],
        "predicted_class": left_pred,
        "confidence": left_conf,
        "heatmap": left_heatmap,
        "image_id": left_uploaded.name,
    }

    return right_data, left_data




def render_eye_result(data, label="Screening Result"):
    """Render the normal single-eye result UI."""
    pred = data["predicted_class"]
    conf = data["confidence"]
    probs = data["probabilities"]

    st.markdown('<div id="sec-assessment" class="section-kicker">03 · AI ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)

    grade_col, confidence_col, screening_col = st.columns(3, gap="medium")

    with grade_col:
        st.markdown(
            f"""<div class="assessment-card">
            <div class="assessment-label">PREDICTED SEVERITY</div>
            <div class="assessment-value">Grade {pred}</div>
            <div class="assessment-subtitle">{CLASS_NAMES[pred]}</div>
            </div>""", unsafe_allow_html=True
        )

    with confidence_col:
        st.markdown(
            f"""<div class="assessment-card">
            <div class="assessment-label">MODEL CONFIDENCE</div>
            <div class="assessment-value">{conf*100:.2f}%</div>
            <div class="assessment-subtitle">Confidence for predicted class</div>
            </div>""", unsafe_allow_html=True
        )
        st.progress(float(conf))

    with screening_col:
        if pred >= 2:
            st.markdown(
                """<div class="status-card referable">
                <div class="status-label">SCREENING ASSESSMENT</div>
                <div class="status-value">REFERABLE DR</div>
                <div class="status-description">Predicted severity falls within Grade 2–4.</div>
                </div>""", unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div class="status-card normal">
                <div class="status-label">SCREENING ASSESSMENT</div>
                <div class="status-value">NON-REFERABLE</div>
                <div class="status-description">Predicted severity falls within Grade 0–1.</div>
                </div>""", unsafe_allow_html=True
            )

    st.divider()
    st.markdown('<div class="section-kicker">CLINICAL CONTEXT</div>', unsafe_allow_html=True)
    st.markdown(f"**Predicted classification:** {GRADE_DESCRIPTIONS[pred]}")
    st.divider()

    st.markdown('<div class="section-kicker">04 · EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Visual AI Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Grad-CAM highlights image regions that contributed to the selected prediction.</div>',
        unsafe_allow_html=True
    )

    tab_original, tab_gradcam, tab_compare = st.tabs(["Original Image", "Grad-CAM", "Side-by-Side"])
    with tab_original:
        st.image(data["image"], use_container_width=True)
    with tab_gradcam:
        st.image(data["heatmap"], use_container_width=True)
        st.caption("Warmer regions indicate stronger model attention.")
    with tab_compare:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("#### Original Retina")
            st.image(data["image"], use_container_width=True)
        with c2:
            st.markdown("#### Grad-CAM Attention")
            st.image(data["heatmap"], use_container_width=True)

    st.divider()
    st.markdown('<div class="section-kicker">05 · MODEL OUTPUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Class Probability Distribution</div>', unsafe_allow_html=True)

    for i in range(5):
        probability = float(probs[0, i])
        a, b = st.columns([6,1], gap="medium")
        with a:
            st.caption(f"Grade {i} · {CLASS_NAMES[i]}")
            st.progress(probability)
        with b:
            st.markdown(f"**{probability*100:.2f}%**")


if screening_mode == "Single Eye":

    if uploaded_file is None:
        st.info("Upload a retinal fundus image above to begin screening.")
        st.markdown(
            """<div id="sec-assessment" class="assessment-card">
            <div class="assessment-label">ANALYSIS PIPELINE // STANDBY</div>
            <div class="assessment-value">Image → AI → Explanation</div>
            <div class="assessment-subtitle">
            Upload a fundus image, receive a five-class severity prediction, and inspect the Grad-CAM regions
            that contributed to the model's decision.
            </div></div>""",
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Analysing retinal image..."):
            single_data = run_eye_analysis(uploaded_file)

        render_eye_result(single_data)

        st.divider()
        st.markdown('<div class="section-kicker">06 · PATIENT REPORT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Generate Screening Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Create a professional PDF containing patient details, AI results, Grad-CAM explanation and model information.</div>',
            unsafe_allow_html=True
        )

        patient_data = {
            "Case ID": case_id,
            "Patient Name": patient_name if patient_name.strip() else "Not provided",
            "Age": str(patient_age),
            "Sex": patient_sex,
            "Screening Date": datetime.now().strftime("%d %b %Y, %H:%M"),
            "Screening Centre": screening_centre if screening_centre.strip() else "Not provided",
            "Eye Examined": eye_examined,
            "Image ID": single_data["image_id"],
            "Diabetes Type": diabetes_type,
            "Diabetes Duration": diabetes_duration if diabetes_duration.strip() else "Not provided",
            "Previous DR": previous_dr,
            "Previous Eye Exam": previous_eye_exam,
        }

        report_bytes = generate_screening_report(
            patient_data, single_data["image"], single_data["heatmap"],
            single_data["probabilities"], single_data["predicted_class"], single_data["confidence"]
        )

        st.download_button(
            label="◈ GENERATE & DOWNLOAD SCREENING REPORT (PDF)",
            data=report_bytes,
            file_name=f"RetinaScope_Report_{case_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:

    if right_eye_file is None or left_eye_file is None:
        st.info("Upload both the Right Eye and Left Eye fundus images to begin dual-eye screening.")
    else:
        with st.spinner("Running AI analysis on both eyes..."):
            right_data, left_data = run_dual_eye_analysis(
                right_eye_file,
                left_eye_file
            )

        st.markdown('<div id="sec-assessment" class="section-kicker">03 · DUAL-EYE ASSESSMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Right Eye vs Left Eye</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Both images are evaluated together by EfficientNet-B0, followed by individual Grad-CAM explanations.</div>',
            unsafe_allow_html=True
        )

        rc, lc = st.columns(2, gap="medium")
        for col, label, data in [(rc, "Right Eye", right_data), (lc, "Left Eye", left_data)]:
            with col:
                st.markdown(f"### ◈ {label}")
                st.image(data["image"], use_container_width=True)
                st.metric(
                    "Predicted Grade",
                    f"Grade {data['predicted_class']} · {CLASS_NAMES[data['predicted_class']]}"
                )
                st.metric("Model Confidence", f"{data['confidence']*100:.2f}%")
                if data["predicted_class"] >= 2:
                    st.error("REFERABLE DR")
                else:
                    st.success("NON-REFERABLE")

        st.divider()
        rgrade = right_data["predicted_class"]
        lgrade = left_data["predicted_class"]
        diff = abs(rgrade - lgrade)

        st.markdown('<div class="section-kicker">04 · INTER-EYE ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Severity Comparison</div>', unsafe_allow_html=True)

        a, b, c = st.columns(3, gap="medium")
        with a:
            st.metric("Right Eye", f"Grade {rgrade}")
        with b:
            st.metric("Left Eye", f"Grade {lgrade}")
        with c:
            st.metric("Grade Difference", str(diff))

        if diff == 0:
            st.success("Both eyes received the same predicted severity grade.")
        else:
            higher = "Right Eye" if rgrade > lgrade else "Left Eye"
            st.warning(f"Different predicted severity levels detected. {higher} has the higher predicted grade.")

        st.caption(
            "Inter-eye comparison describes model outputs for the two uploaded images. "
            "It is not a clinical diagnosis of disease asymmetry."
        )

        st.divider()
        st.markdown('<div class="section-kicker">05 · DUAL-EYE EXPLAINABILITY</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Model Attention — Both Eyes</div>', unsafe_allow_html=True)

        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown("#### Right Eye · Grad-CAM")
            st.image(right_data["heatmap"], use_container_width=True)
        with b:
            st.markdown("#### Left Eye · Grad-CAM")
            st.image(left_data["heatmap"], use_container_width=True)

        st.caption(
            "Grad-CAM highlights regions that contributed to each selected prediction. "
            "Highlighted regions represent model attention, not confirmed lesions."
        )

        st.divider()
        st.markdown('<div class="section-kicker">06 · MODEL OUTPUT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Probability Comparison</div>', unsafe_allow_html=True)

        for i in range(5):
            rp = float(right_data["probabilities"][0,i])
            lp = float(left_data["probabilities"][0,i])
            st.markdown(f"**Grade {i} · {CLASS_NAMES[i]}**")
            a, b = st.columns(2, gap="medium")
            with a:
                st.caption(f"Right Eye · {rp*100:.2f}%")
                st.progress(rp)
            with b:
                st.caption(f"Left Eye · {lp*100:.2f}%")
                st.progress(lp)

        st.divider()
        st.markdown('<div class="section-kicker">06B · PATIENT REPORT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Generate Dual-Eye Screening Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Create a PDF containing both-eye results, inter-eye comparison, Grad-CAM explanations and probability distributions.</div>',
            unsafe_allow_html=True
        )

        dual_patient_data = {
            "Case ID": case_id,
            "Patient Name": patient_name if patient_name.strip() else "Not provided",
            "Age": str(patient_age),
            "Sex": patient_sex,
            "Screening Date": datetime.now().strftime("%d %b %Y, %H:%M"),
            "Screening Centre": screening_centre if screening_centre.strip() else "Not provided",
            "Right Image ID": right_data["image_id"],
            "Left Image ID": left_data["image_id"],
            "Diabetes Type": diabetes_type,
            "Diabetes Duration": diabetes_duration if diabetes_duration.strip() else "Not provided",
            "Previous DR": previous_dr,
            "Previous Eye Exam": previous_eye_exam,
        }

        dual_report_bytes = generate_dual_eye_report(dual_patient_data, right_data, left_data)

        st.download_button(
            label="◈ GENERATE & DOWNLOAD DUAL-EYE REPORT (PDF)",
            data=dual_report_bytes,
            file_name=f"RetinaScope_DualEye_Report_{case_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div id="sec-validation" class="section-kicker">07 · MODEL PERFORMANCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">Validation Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Performance measured on the held-out APTOS validation split.</div>',
    unsafe_allow_html=True
)

perf1, perf2, perf3, perf4 = st.columns(
    4,
    gap="medium"
)

with perf1:
    st.metric(
        "5-Class Accuracy",
        "80.22%"
    )

with perf2:
    st.metric(
        "Macro F1",
        "65.03%"
    )

with perf3:
    st.metric(
        "Referable Sensitivity",
        "90.27%"
    )

with perf4:
    st.metric(
        "Referable Specificity",
        "92.18%"
    )


# ============================================================
# RESOURCE PLANNING
# ============================================================

st.divider()

st.markdown(
    '<div id="sec-planning" class="section-kicker">08 · RESOURCE PLANNING</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-heading">District Screening Capacity</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">'
    'District-level planning view using Census 2011 population data and '
    'transparent prototype estimates for screening demand and capacity.'
    '</div>',
    unsafe_allow_html=True
)

try:
    planning_df = load_planning_data()

    state_col, priority_col = st.columns(2, gap="medium")

    with state_col:
        states = ["All States"] + sorted(
            planning_df["state"].dropna().unique().tolist()
        )
        selected_state = st.selectbox(
            "State", states, key="planning_state"
        )

    with priority_col:
        priorities = ["All Priorities", "Critical", "High", "Moderate", "Low"]
        selected_priority = st.selectbox(
            "Priority", priorities, key="planning_priority"
        )

    filtered = planning_df.copy()

    if selected_state != "All States":
        filtered = filtered[filtered["state"] == selected_state]

    if selected_priority != "All Priorities":
        filtered = filtered[filtered["priority"] == selected_priority]

    # Summary
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.metric("Districts", f"{len(filtered):,}")

    with c2:
        st.metric(
            "Annual Demand",
            f"{int(filtered['estimated_annual_screening_demand'].sum()):,}"
        )

    with c3:
        st.metric(
            "Capacity Gap",
            f"{int(filtered['capacity_gap'].sum()):,}"
        )

    with c4:
        st.metric(
            "High / Critical",
            f"{int(filtered['priority'].isin(['High', 'Critical']).sum()):,}"
        )

    # ---------------- TABLE ----------------
    # Render the planning table BEFORE the map so district data remains
    # immediately visible even if browser-side map rendering is slow.
    st.markdown("#### Priority Districts")

    table = filtered[
        [
            "state",
            "district",
            "estimated_annual_screening_demand",
            "estimated_screening_capacity",
            "capacity_gap",
            "priority",
        ]
    ].sort_values(
        "capacity_gap",
        ascending=False
    )

    st.dataframe(
        table.rename(columns={
            "state": "State",
            "district": "District",
            "estimated_annual_screening_demand": "Annual Demand",
            "estimated_screening_capacity": "Capacity",
            "capacity_gap": "Capacity Gap",
            "priority": "Priority",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------- MAP ----------------
    try:
        map_df = build_map_data(selected_state)

        if selected_priority != "All Priorities":
            map_df = map_df[
                map_df["priority"] == selected_priority
            ].copy()

        if not map_df.empty:
            geojson = map_df.__geo_interface__

            priority_order = [
                "Low",
                "Moderate",
                "High",
                "Critical"
            ]

            map_df["priority"] = pd.Categorical(
                map_df["priority"],
                categories=priority_order,
                ordered=True,
            )

            fig = px.choropleth(
                map_df,
                geojson=geojson,
                locations="map_id",
                featureidkey="properties.map_id",
                color="priority",
                category_orders={"priority": priority_order},
                color_discrete_map={
                    "Low": "#00f59b",
                    "Moderate": "#e5c158",
                    "High": "#ff8c00",
                    "Critical": "#ff3b5c",
                },
                hover_name="district",
                hover_data={
                    "state": True,
                    "priority": True,
                    "capacity_gap": ":,",
                    "estimated_annual_screening_demand": ":,",
                    "estimated_screening_capacity": ":,",
                    "priority_score": ":.1f",
                    "map_id": False,
                },
                labels={
                    "priority": "Priority",
                    "capacity_gap": "Capacity Gap",
                    "estimated_annual_screening_demand": "Annual Demand",
                    "estimated_screening_capacity": "Capacity",
                    "priority_score": "Priority Score",
                },
            )

            fig.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="rgba(0,0,0,0)",
            )

            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Rajdhani, sans-serif"),
                legend=dict(
                    title="Priority",
                    orientation="h",
                    yanchor="bottom",
                    y=1.01,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )
        else:
            st.info("No mapped districts match the selected filters.")

    except Exception as map_error:
        st.warning(
            f"District map unavailable: {map_error}"
        )

    st.caption(
        "Planning estimates are prototype assumptions derived from "
        "district-level Census 2011 population data; they are not observed "
        "healthcare capacity statistics."
    )

except Exception:
    st.error(
        "Planning data could not be loaded. Make sure planning_data.csv "
        "is in the project root."
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.warning(
    "Research prototype only. The AI output and Grad-CAM "
    "visualization are not a clinical diagnosis and should "
    "not replace evaluation by a qualified healthcare professional. "
    "Reported performance is based on an internal APTOS "
    "validation split and does not establish clinical validity."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="app-footer">

<b>RETINASCOPE // AETHERIAL RENAISSANCE EDITION</b> · EXPLAINABLE AI FOR DIABETIC RETINOPATHY SCREENING

<br>

EfficientNet-B0 · Grad-CAM · APTOS 2019 · District Cartography

<br>

Research & Demonstration Prototype

</div>
""",
    unsafe_allow_html=True
)