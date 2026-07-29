"""
Seed script — populates supply_chain_demo with inventory, suppliers, 90 days of
consumption history, and a historical order archive (used by Vector Search).

Also creates:
  - Atlas Search index on `suppliers`  (full-text search over supplier notes)
  - Atlas Vector Search index on `order_history` (semantic similarity search)

Run once before starting the simulator:
    python data/seed.py
"""

import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from bson import ObjectId
from pymongo import MongoClient, ASCENDING

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.alerts import build_reorder_alert, insert_reorder_alert
from agent.embeddings import embeddings as _embeddings, EMBEDDING_DIMS

load_dotenv()

client = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=30_000,
    connectTimeoutMS=20_000,
    socketTimeoutMS=45_000,
)
db = client["supply_chain_demo"]

# Grove API gateway (Azure-style api-key auth)
_GROVE_API_KEY  = os.environ["GROVE_API_KEY"]
_GROVE_BASE_URL = os.environ["GROVE_API_BASE_URL"]


# ---------------------------------------------------------------------------
# Master SKU catalogue
# ---------------------------------------------------------------------------
SKUS = [
    {
        # Above reorder point — healthy stock
        "sku": "MED-2041",
        "name": "Amoxicillin 500mg Capsules",
        "location": "DC-Ohio",
        "on_hand": 320,
        "on_order": 0,
        "reorder_point": 200,
        "safety_stock": 80,
        "unit_cost": 0.45,
        "category": "pharmaceutical",
    },
    {
        # Above reorder point — healthy stock
        "sku": "MED-3017",
        "name": "Insulin Glargine",
        "location": "DC-Texas",
        "on_hand": 1350,
        "on_order": 0,
        "reorder_point": 1000,
        "safety_stock": 300,
        "unit_cost": 12.50,
        "category": "pharmaceutical",
    },
    {
        # Below reorder point (~55%) — triggers alert immediately (1 of 3)
        "sku": "SURG-0084",
        "name": "Nitrile Gloves (box)",
        "location": "DC-Ohio",
        "on_hand": 55,
        "on_order": 0,
        "reorder_point": 100,
        "safety_stock": 30,
        "unit_cost": 8.95,
        "category": "surgical",
    },
    {
        # Above reorder point — healthy stock
        "sku": "SURG-1122",
        "name": "IV Bags 1L",
        "location": "DC-Texas",
        "on_hand": 680,
        "on_order": 0,
        "reorder_point": 500,
        "safety_stock": 150,
        "unit_cost": 2.10,
        "category": "surgical",
    },
    {
        # Above reorder point — intentionally well-stocked
        "sku": "MED-4490",
        "name": "Metformin 1000mg",
        "location": "DC-Ohio",
        "on_hand": 2100,
        "on_order": 0,
        "reorder_point": 1500,
        "safety_stock": 500,
        "unit_cost": 0.18,
        "category": "pharmaceutical",
    },
    {
        # Below reorder point (~42%) — triggers alert immediately (2 of 3)
        "sku": "MED-5502",
        "name": "Vancomycin 1g IV",
        "location": "DC-California",
        "on_hand": 85,
        "on_order": 0,
        "reorder_point": 200,
        "safety_stock": 60,
        "unit_cost": 8.50,
        "category": "pharmaceutical",
    },
    {
        # Below reorder point (~75%) — triggers alert immediately (3 of 3)
        "sku": "MED-6201",
        "name": "Heparin Sodium 5000U/mL",
        "location": "DC-Texas",
        "on_hand": 450,
        "on_order": 0,
        "reorder_point": 600,
        "safety_stock": 180,
        "unit_cost": 3.20,
        "category": "pharmaceutical",
    },
    {
        # Above reorder point — healthy stock
        "sku": "DIAG-0331",
        "name": "Rapid COVID/Flu Combo Test Kit",
        "location": "DC-Ohio",
        "on_hand": 680,
        "on_order": 0,
        "reorder_point": 500,
        "safety_stock": 150,
        "unit_cost": 4.75,
        "category": "diagnostic",
    },
    {
        # Above reorder point — healthy stock
        "sku": "SURG-2244",
        "name": "N95 Respirator Mask",
        "location": "DC-California",
        "on_hand": 580,
        "on_order": 0,
        "reorder_point": 400,
        "safety_stock": 120,
        "unit_cost": 2.85,
        "category": "surgical",
    },
    {
        # Above reorder point — healthy stock
        "sku": "LAB-0112",
        "name": "Aerobic Blood Culture Bottle",
        "location": "DC-Texas",
        "on_hand": 580,
        "on_order": 0,
        "reorder_point": 400,
        "safety_stock": 100,
        "unit_cost": 6.20,
        "category": "laboratory",
    },
]

# ---------------------------------------------------------------------------
# Supplier catalogue — 2 per SKU.
# `notes` is free-text: indexed by Atlas Search for capability matching.
# ---------------------------------------------------------------------------
SUPPLIERS = [
    # MED-2041
    {
        "sku": "MED-2041", "supplier_name": "PharmaCo Ltd", "supplier_id": "SUP-001",
        "lead_time_days": 5, "moq": 500, "unit_price": 0.42,
        "fill_rate_pct": 97.2, "on_time_delivery_pct": 94.1, "fda_registered": True,
        "notes": (
            "Specializes in beta-lactam antibiotics and broad-spectrum oral pharmaceuticals. "
            "FDA-registered with GDP compliance. Offers expedited 48-hour rush delivery for "
            "critical low-stock situations. Strong cold-chain and temperature-controlled "
            "distribution capability for pharmaceutical products."
        ),
    },
    {
        "sku": "MED-2041", "supplier_name": "MediSource Inc", "supplier_id": "SUP-002",
        "lead_time_days": 7, "moq": 250, "unit_price": 0.44,
        "fill_rate_pct": 92.0, "on_time_delivery_pct": 89.5, "fda_registered": True,
        "notes": (
            "General pharmaceutical distributor with broad antibiotic and oral medication catalog. "
            "Competitive pricing on lower minimum order quantities. Suitable for routine "
            "replenishment cycles. Standard warehouse-to-dock delivery with no expedite option."
        ),
    },
    # MED-3017
    {
        "sku": "MED-3017", "supplier_name": "BioPharm Global", "supplier_id": "SUP-003",
        "lead_time_days": 10, "moq": 200, "unit_price": 11.80,
        "fill_rate_pct": 98.5, "on_time_delivery_pct": 96.0, "fda_registered": True,
        "notes": (
            "Biologics and insulin specialist with strict cold-chain and 2-8°C temperature "
            "controlled logistics. Handles temperature-sensitive pharmaceutical injectables. "
            "ISO 13485 certified. Preferred partner for hospital-grade insulin supply chains. "
            "Dedicated account managers for healthcare distribution networks."
        ),
    },
    {
        # NOT FDA-registered — smaller secondary supplier, no wholesale drug distributor licence.
        # The regulatory filter blocks this supplier for pharmaceutical SKUs.
        "sku": "MED-3017", "supplier_name": "Insulin Direct", "supplier_id": "SUP-004",
        "lead_time_days": 14, "moq": 100, "unit_price": 12.00,
        "fill_rate_pct": 88.0, "on_time_delivery_pct": 85.0, "fda_registered": False,
        "notes": (
            "Dedicated insulin and diabetes medication supplier. Lower fill rates and longer "
            "lead times make this supplier suitable for supplemental orders when the primary "
            "supplier is back-ordered. Smaller MOQ allows topping up without over-committing capital."
        ),
    },
    # SURG-0084
    {
        "sku": "SURG-0084", "supplier_name": "SafeGlove Co", "supplier_id": "SUP-005",
        "lead_time_days": 3, "moq": 50, "unit_price": 8.50,
        "fill_rate_pct": 99.0, "on_time_delivery_pct": 97.5, "fda_registered": True,
        "notes": (
            "PPE and surgical glove manufacturer with domestic warehouse for rapid fulfillment. "
            "Specializes in nitrile, latex-free, and sterile surgical gloves for hospital and "
            "clinical settings. Fastest lead time in category. FDA 510(k) cleared for surgical use. "
            "Reliable partner for urgent and emergency PPE stock replenishment."
        ),
    },
    {
        "sku": "SURG-0084", "supplier_name": "MedSupply Direct", "supplier_id": "SUP-006",
        "lead_time_days": 5, "moq": 100, "unit_price": 8.20,
        "fill_rate_pct": 94.0, "on_time_delivery_pct": 91.0, "fda_registered": True,
        "notes": (
            "Medical consumables distributor covering gloves, masks, gowns, and general PPE. "
            "Bulk pricing available for high-volume orders. Standard warehouse-to-dock delivery. "
            "Consistent product quality with lot traceability documentation."
        ),
    },
    # SURG-1122
    {
        "sku": "SURG-1122", "supplier_name": "IV Solutions LLC", "supplier_id": "SUP-007",
        "lead_time_days": 4, "moq": 200, "unit_price": 1.95,
        "fill_rate_pct": 96.5, "on_time_delivery_pct": 93.0, "fda_registered": True,
        "notes": (
            "IV fluid and parenteral solution specialist maintaining emergency reserve inventory "
            "for hospital networks. Can fulfill urgent critical orders within 24 hours for "
            "established accounts. Covers saline, dextrose, and lactated Ringer's formulations. "
            "Preferred vendor for surgical and critical care IV supply chains."
        ),
    },
    {
        "sku": "SURG-1122", "supplier_name": "ClearFlow Med", "supplier_id": "SUP-008",
        "lead_time_days": 6, "moq": 500, "unit_price": 1.90,
        "fill_rate_pct": 91.0, "on_time_delivery_pct": 88.0, "fda_registered": True,
        "notes": (
            "High-volume IV bags and infusion equipment supplier. Best price-per-unit at large "
            "order quantities. Suitable for planned procurement and standing orders. Full batch "
            "traceability and sterility documentation. Longer lead time requires advance planning."
        ),
    },
    # MED-4490
    {
        "sku": "MED-4490", "supplier_name": "GeneriPharma", "supplier_id": "SUP-009",
        "lead_time_days": 7, "moq": 1000, "unit_price": 0.17,
        "fill_rate_pct": 95.0, "on_time_delivery_pct": 92.0, "fda_registered": True,
        "notes": (
            "Generic oral medication manufacturer with high-volume metformin and diabetes drug "
            "production lines. Cost-effective for large standing orders. FDA ANDA approved "
            "manufacturing facility. Reliable partner for planned pharmaceutical replenishment."
        ),
    },
    {
        # NOT FDA-registered — bulk wholesale broker without licensed distributor status.
        # The regulatory filter blocks this supplier for pharmaceutical SKUs.
        "sku": "MED-4490", "supplier_name": "BulkRx Wholesale", "supplier_id": "SUP-010",
        "lead_time_days": 10, "moq": 2000, "unit_price": 0.15,
        "fill_rate_pct": 89.0, "on_time_delivery_pct": 86.0, "fda_registered": False,
        "notes": (
            "Wholesale pharmaceutical distributor offering lowest unit pricing at very high MOQ. "
            "Long lead times require integration into quarterly procurement planning cycles. "
            "Best suited for stable, predictable demand items with ample safety stock on hand."
        ),
    },
    # MED-5502
    {
        "sku": "MED-5502", "supplier_name": "NovaBiotics Inc", "supplier_id": "SUP-011",
        "lead_time_days": 3, "moq": 50, "unit_price": 8.25,
        "fill_rate_pct": 98.0, "on_time_delivery_pct": 95.5, "fda_registered": True,
        "notes": (
            "Specialty IV antibiotic manufacturer with 24-hour emergency fulfillment for vancomycin "
            "and glycopeptide antibiotics. FDA-registered and GDP-compliant with lot traceability. "
            "Dedicated emergency stock reserve maintained for MRSA and critical care antibiotic "
            "programs. Preferred partner for hospital infection control procurement."
        ),
    },
    {
        "sku": "MED-5502", "supplier_name": "AntibioCare Pharma", "supplier_id": "SUP-012",
        "lead_time_days": 6, "moq": 25, "unit_price": 8.75,
        "fill_rate_pct": 91.0, "on_time_delivery_pct": 88.0, "fda_registered": True,
        "notes": (
            "Hospital pharmacy supplier specializing in IV-formulated antibiotics and antifungals. "
            "Low MOQ suitable for smaller top-up orders between primary supplier deliveries. "
            "Standard 6-day lead time with no expedite option. Good backup for broad-spectrum "
            "and critical care antibiotic supply when primary supplier faces capacity constraints."
        ),
    },
    # MED-6201
    {
        "sku": "MED-6201", "supplier_name": "CoagPharma Ltd", "supplier_id": "SUP-013",
        "lead_time_days": 4, "moq": 200, "unit_price": 3.10,
        "fill_rate_pct": 97.0, "on_time_delivery_pct": 94.0, "fda_registered": True,
        "notes": (
            "Anticoagulant and parenteral medication specialist with robust heparin supply chain. "
            "Maintains inventory buffer for surgical program support and peri-operative care. "
            "GDP-certified with cold-chain capability for biologics. Supplies hospital cardiac, "
            "surgical and ICU units with fast order processing for thrombosis prevention protocols."
        ),
    },
    {
        # NOT FDA-registered — secondary broker without licensed wholesale distributor status.
        # The regulatory filter blocks this supplier for pharmaceutical SKUs.
        "sku": "MED-6201", "supplier_name": "AnticoagDirect", "supplier_id": "SUP-014",
        "lead_time_days": 7, "moq": 100, "unit_price": 3.25,
        "fill_rate_pct": 90.0, "on_time_delivery_pct": 87.0, "fda_registered": False,
        "notes": (
            "Secondary heparin and anticoagulant distributor. Longer lead time offset by low MOQ "
            "and competitive pricing on smaller orders. Suitable as backup when primary supplier "
            "faces shortage or capacity constraints. No expedite option available."
        ),
    },
    # DIAG-0331
    {
        "sku": "DIAG-0331", "supplier_name": "QuickDiag Labs", "supplier_id": "SUP-015",
        "lead_time_days": 2, "moq": 200, "unit_price": 4.50,
        "fill_rate_pct": 96.0, "on_time_delivery_pct": 93.0, "fda_registered": True,
        "notes": (
            "Rapid diagnostic test kit manufacturer with domestic warehouse for fast-turnaround "
            "fulfillment. Specializes in combo respiratory panel kits covering influenza A/B and "
            "COVID-19 detection. Point-of-care FDA EUA cleared. Maintains surge inventory for "
            "seasonal flu and outbreak response. Fastest lead time in the diagnostic category."
        ),
    },
    {
        "sku": "DIAG-0331", "supplier_name": "TestBridge Inc", "supplier_id": "SUP-016",
        "lead_time_days": 5, "moq": 500, "unit_price": 4.60,
        "fill_rate_pct": 88.0, "on_time_delivery_pct": 85.0, "fda_registered": True,
        "notes": (
            "Bulk diagnostic test kit distributor with strong pricing at high volumes. Suited for "
            "seasonal and pandemic preparedness procurement planning. Longer lead time requires "
            "advance ordering during peak respiratory illness season. Full regulatory documentation "
            "including EUA certificates and lot traceability provided on all orders."
        ),
    },
    # SURG-2244
    {
        "sku": "SURG-2244", "supplier_name": "RespiraTech PPE", "supplier_id": "SUP-017",
        "lead_time_days": 3, "moq": 200, "unit_price": 2.75,
        "fill_rate_pct": 98.5, "on_time_delivery_pct": 96.0, "fda_registered": True,
        "notes": (
            "NIOSH-approved N95 respirator manufacturer with domestic production for rapid "
            "fulfillment. Specializes in healthcare-grade respiratory protection for surgical "
            "and isolation settings. FDA 510(k) cleared. Emergency PPE surge stock maintained "
            "for infection control and outbreak preparedness. Fastest lead time in category."
        ),
    },
    {
        "sku": "SURG-2244", "supplier_name": "N95Shield Supply", "supplier_id": "SUP-018",
        "lead_time_days": 5, "moq": 500, "unit_price": 2.65,
        "fill_rate_pct": 92.0, "on_time_delivery_pct": 89.0, "fda_registered": True,
        "notes": (
            "High-volume N95 and surgical mask distributor. Best unit pricing at large order "
            "quantities for planned replenishment and strategic stockpile procurement. Full NIOSH "
            "certification and batch documentation available. Standard warehouse-to-dock delivery "
            "with no expedite option. Preferred for non-urgent bulk PPE orders."
        ),
    },
    # LAB-0112
    {
        "sku": "LAB-0112", "supplier_name": "LabCore Supplies", "supplier_id": "SUP-019",
        "lead_time_days": 3, "moq": 100, "unit_price": 6.00,
        "fill_rate_pct": 97.5, "on_time_delivery_pct": 95.0, "fda_registered": True,
        "notes": (
            "Clinical laboratory supply specialist with rapid order fulfillment for blood culture "
            "media and microbiology consumables. Covers aerobic, anaerobic and fungal culture "
            "bottle formats. Maintains emergency stock for bacteremia and sepsis workup programs. "
            "ISO 13485 certified. Preferred supplier for critical diagnostic laboratory supply."
        ),
    },
    {
        "sku": "LAB-0112", "supplier_name": "ClinPath Direct", "supplier_id": "SUP-020",
        "lead_time_days": 6, "moq": 200, "unit_price": 5.80,
        "fill_rate_pct": 90.0, "on_time_delivery_pct": 87.5, "fda_registered": True,
        "notes": (
            "Clinical pathology consumables distributor. Competitive pricing on bulk blood culture "
            "and microbiology supply orders. Suitable for routine planned replenishment cycles. "
            "Longer lead time requires advance ordering. Full lot traceability and sterility "
            "documentation provided. Good backup supplier for laboratory supply continuity."
        ),
    },
]

# ---------------------------------------------------------------------------
# Average daily consumption per SKU — used to generate 90-day history
# ---------------------------------------------------------------------------
AVG_DAILY = {
    "MED-2041":  45,
    "MED-3017": 120,
    "SURG-0084": 18,
    "SURG-1122":  60,
    "MED-4490":  90,
    "MED-5502":  25,
    "MED-6201":  55,
    "DIAG-0331": 90,
    "SURG-2244": 70,
    "LAB-0112":  35,
}

# SKUs that receive a linearly rising demand trend in their consumption history.
# Days 1–60 are seeded flat; days 61–90 ramp from 1.0× to 2.0× the base average.
# This ensures get_consumption_trend() reliably returns trend="increasing" for
# these SKUs, exercising the confidence-downgrade path in the agent prompt.
# One SKU per category so all three confidence-downgrade scenarios are visible:
#   MED-3017  → pharmaceutical (insulin — chronic disease demand surge)
#   SURG-0084 → surgical       (nitrile gloves — seasonal hospital volume)
#   DIAG-0331 → diagnostic     (COVID/flu test kits — respiratory season)
RISING_TREND_SKUS = {"MED-3017", "SURG-0084", "DIAG-0331"}

# How many of the most-recent days carry the rising ramp (must be < 90).
_RAMP_DAYS = 30

LOCATION_BY_SKU  = {s["sku"]: s["location"]  for s in SKUS}
CATEGORY_BY_SKU  = {s["sku"]: s["category"]  for s in SKUS}

# ---------------------------------------------------------------------------
# Historical order archive — used to seed `order_history` collection.
# Each record's `rationale` field gets embedded; Vector Search uses those
# embeddings to find precedent for new alerts.
# ---------------------------------------------------------------------------
ORDER_HISTORY = [
    # MED-2041 — various historical scenarios
    {
        "sku": "MED-2041", "location": "DC-Ohio",
        "supplier_id": "SUP-001", "supplier_name": "PharmaCo Ltd",
        "quantity": 1500, "unit_price": 0.42, "total_cost": 630.00,
        "days_of_stock_at_order": 2.1, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 11, 5, 10, 30, tzinfo=timezone.utc),
        "rationale": (
            "Amoxicillin stock critically low at 2.1 days remaining with an increasing "
            "consumption trend driven by seasonal respiratory illness surge. Ordered 1500 "
            "units from PharmaCo Ltd to cover 30 days of demand plus safety buffer. "
            "Expedited delivery requested due to urgency."
        ),
    },
    {
        "sku": "MED-2041", "location": "DC-Ohio",
        "supplier_id": "SUP-001", "supplier_name": "PharmaCo Ltd",
        "quantity": 1000, "unit_price": 0.42, "total_cost": 420.00,
        "days_of_stock_at_order": 4.5, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 9, 12, 14, 0, tzinfo=timezone.utc),
        "rationale": (
            "Routine reorder for Amoxicillin 500mg at DC-Ohio. Stock at 4.5 days with stable "
            "consumption. Ordered standard 1000 units from preferred supplier PharmaCo Ltd "
            "to maintain buffer ahead of a scheduled distribution centre audit."
        ),
    },
    {
        "sku": "MED-2041", "location": "DC-Ohio",
        "supplier_id": "SUP-002", "supplier_name": "MediSource Inc",
        "quantity": 500, "unit_price": 0.44, "total_cost": 220.00,
        "days_of_stock_at_order": 3.8, "trend_at_order": "decreasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 7, 20, 9, 0, tzinfo=timezone.utc),
        "rationale": (
            "Supplemental Amoxicillin order via MediSource due to primary supplier capacity "
            "constraints. Consumption trend is decreasing post-infection season. Ordered "
            "minimum 500 units to bridge the gap while PharmaCo restocked their warehouse."
        ),
    },
    # MED-3017 historical orders
    {
        "sku": "MED-3017", "location": "DC-Texas",
        "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
        "quantity": 4000, "unit_price": 11.80, "total_cost": 47200.00,
        "days_of_stock_at_order": 1.8, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 12, 3, 8, 0, tzinfo=timezone.utc),
        "rationale": (
            "Insulin Glargine critically low at 1.8 days. Consumption spiking due to new "
            "hospital contract adding 200 daily doses. Emergency order of 4000 units placed "
            "with BioPharm Global for cold-chain delivery. High-value order justifies "
            "premium supplier given strict temperature requirements."
        ),
    },
    {
        "sku": "MED-3017", "location": "DC-Texas",
        "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
        "quantity": 3600, "unit_price": 11.80, "total_cost": 42480.00,
        "days_of_stock_at_order": 5.2, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 10, 18, 11, 30, tzinfo=timezone.utc),
        "rationale": (
            "Standard monthly Insulin Glargine replenishment for DC-Texas. Stable consumption "
            "at approximately 120 units per day. BioPharm Global selected for cold-chain "
            "compliance and 98.5% fill rate. 30-day coverage plus safety stock buffer ordered."
        ),
    },
    {
        "sku": "MED-3017", "location": "DC-Texas",
        "supplier_id": "SUP-004", "supplier_name": "Insulin Direct",
        "quantity": 400, "unit_price": 12.00, "total_cost": 4800.00,
        "days_of_stock_at_order": 3.0, "trend_at_order": "stable",
        "outcome": "delayed",
        "ordered_at": datetime(2024, 8, 5, 15, 0, tzinfo=timezone.utc),
        "rationale": (
            "Backup insulin order via Insulin Direct while BioPharm Global experienced a "
            "cold-chain logistics disruption. Secondary supplier selected despite longer lead "
            "time to ensure continuity of supply. Smaller quantity to minimise exposure."
        ),
    },
    # SURG-0084 historical orders
    {
        "sku": "SURG-0084", "location": "DC-Ohio",
        "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
        "quantity": 600, "unit_price": 8.50, "total_cost": 5100.00,
        "days_of_stock_at_order": 1.2, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 11, 28, 7, 0, tzinfo=timezone.utc),
        "rationale": (
            "Nitrile gloves at critical level of 1.2 days remaining. Surgical procedure volume "
            "surged due to a regional emergency, driving increased PPE consumption. Expedited "
            "order placed with SafeGlove Co for fastest 3-day lead time. Patient safety priority."
        ),
    },
    {
        "sku": "SURG-0084", "location": "DC-Ohio",
        "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
        "quantity": 350, "unit_price": 8.50, "total_cost": 2975.00,
        "days_of_stock_at_order": 4.0, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 9, 30, 13, 0, tzinfo=timezone.utc),
        "rationale": (
            "Routine glove replenishment at DC-Ohio. Consumption stable at approximately 18 "
            "boxes per day. SafeGlove Co preferred for 99% fill rate and rapid 3-day fulfillment. "
            "Ordered 350 units to restore full safety stock position."
        ),
    },
    {
        "sku": "SURG-0084", "location": "DC-Ohio",
        "supplier_id": "SUP-006", "supplier_name": "MedSupply Direct",
        "quantity": 200, "unit_price": 8.20, "total_cost": 1640.00,
        "days_of_stock_at_order": 3.5, "trend_at_order": "decreasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 6, 14, 10, 0, tzinfo=timezone.utc),
        "rationale": (
            "Supplemental glove order from MedSupply Direct for bulk cost savings. Consumption "
            "trend declining post-procedure peak. Lower price point at 200+ unit order offset "
            "the slightly longer lead time for this non-urgent top-up."
        ),
    },
    # SURG-1122 historical orders
    {
        "sku": "SURG-1122", "location": "DC-Texas",
        "supplier_id": "SUP-007", "supplier_name": "IV Solutions LLC",
        "quantity": 2200, "unit_price": 1.95, "total_cost": 4290.00,
        "days_of_stock_at_order": 2.5, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 12, 10, 9, 30, tzinfo=timezone.utc),
        "rationale": (
            "IV Bags stock falling fast at 2.5 days due to ICU expansion adding 30 new beds. "
            "Ordered 2200 units from IV Solutions LLC who maintain emergency reserves. "
            "Chose over ClearFlow despite higher unit price to guarantee 24-hour expedite "
            "delivery given the patient care criticality."
        ),
    },
    {
        "sku": "SURG-1122", "location": "DC-Texas",
        "supplier_id": "SUP-008", "supplier_name": "ClearFlow Med",
        "quantity": 2000, "unit_price": 1.90, "total_cost": 3800.00,
        "days_of_stock_at_order": 6.0, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 10, 5, 14, 0, tzinfo=timezone.utc),
        "rationale": (
            "Planned IV bag replenishment with adequate lead time available. ClearFlow Med "
            "selected for best price-per-unit at 2000+ quantity with 6 days of stock providing "
            "comfortable buffer. Standard delivery acceptable given no urgency."
        ),
    },
    {
        "sku": "SURG-1122", "location": "DC-Texas",
        "supplier_id": "SUP-007", "supplier_name": "IV Solutions LLC",
        "quantity": 1800, "unit_price": 1.95, "total_cost": 3510.00,
        "days_of_stock_at_order": 4.2, "trend_at_order": "decreasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 8, 22, 11, 0, tzinfo=timezone.utc),
        "rationale": (
            "IV bag reorder at 4.2 days with consumption trend decreasing post-surgical peak. "
            "IV Solutions LLC chosen for reliable fill rate and faster lead time over ClearFlow. "
            "Quantity set to cover 30 days at lower projected demand plus safety stock."
        ),
    },
    # MED-4490 (above reorder — only historical orders from before seeding)
    {
        "sku": "MED-4490", "location": "DC-Ohio",
        "supplier_id": "SUP-009", "supplier_name": "GeneriPharma",
        "quantity": 3000, "unit_price": 0.17, "total_cost": 510.00,
        "days_of_stock_at_order": 8.0, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 11, 15, 10, 0, tzinfo=timezone.utc),
        "rationale": (
            "Quarterly Metformin replenishment at DC-Ohio. Stable chronic disease medication "
            "with predictable demand. GeneriPharma selected for cost efficiency at scale. "
            "Ordered well in advance of reorder point to align with procurement cycle."
        ),
    },
    # MED-5502 historical orders
    {
        "sku": "MED-5502", "location": "DC-California",
        "supplier_id": "SUP-011", "supplier_name": "NovaBiotics Inc",
        "quantity": 300, "unit_price": 8.25, "total_cost": 2475.00,
        "days_of_stock_at_order": 1.4, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 12, 8, 6, 0, tzinfo=timezone.utc),
        "rationale": (
            "Vancomycin 1g IV critically low at 1.4 days of stock remaining with consumption "
            "surging due to an MRSA cluster in the ICU requiring extended antibiotic courses. "
            "Emergency order placed with NovaBiotics Inc for 24-hour fulfillment from their "
            "dedicated infection-control reserve. Patient safety and treatment continuity critical."
        ),
    },
    {
        "sku": "MED-5502", "location": "DC-California",
        "supplier_id": "SUP-011", "supplier_name": "NovaBiotics Inc",
        "quantity": 200, "unit_price": 8.25, "total_cost": 1650.00,
        "days_of_stock_at_order": 5.5, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 10, 22, 11, 0, tzinfo=timezone.utc),
        "rationale": (
            "Routine Vancomycin 1g IV replenishment at DC-California. Stable consumption at "
            "approximately 25 vials per day. NovaBiotics Inc selected for 98% fill rate and "
            "3-day lead time. Order sized for 30-day coverage plus safety stock buffer."
        ),
    },
    {
        "sku": "MED-5502", "location": "DC-California",
        "supplier_id": "SUP-012", "supplier_name": "AntibioCare Pharma",
        "quantity": 75, "unit_price": 8.75, "total_cost": 656.25,
        "days_of_stock_at_order": 3.0, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 8, 14, 15, 0, tzinfo=timezone.utc),
        "rationale": (
            "Supplemental Vancomycin top-up via AntibioCare Pharma while NovaBiotics faced a "
            "temporary production shortage. Low MOQ of 25 allowed a targeted bridge order. "
            "Higher unit price accepted to maintain safety stock without over-committing capital."
        ),
    },
    # MED-6201 historical orders
    {
        "sku": "MED-6201", "location": "DC-Texas",
        "supplier_id": "SUP-013", "supplier_name": "CoagPharma Ltd",
        "quantity": 1800, "unit_price": 3.10, "total_cost": 5580.00,
        "days_of_stock_at_order": 2.0, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 11, 20, 8, 30, tzinfo=timezone.utc),
        "rationale": (
            "Heparin Sodium stock at 2.0 days with consumption rising sharply ahead of a "
            "high-volume elective surgical week. Ordered 1800 units from CoagPharma to cover "
            "peri-operative anticoagulation demand plus safety buffer. Urgency justified "
            "expedited processing given critical surgical program dependency."
        ),
    },
    {
        "sku": "MED-6201", "location": "DC-Texas",
        "supplier_id": "SUP-013", "supplier_name": "CoagPharma Ltd",
        "quantity": 1600, "unit_price": 3.10, "total_cost": 4960.00,
        "days_of_stock_at_order": 6.5, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 9, 9, 10, 0, tzinfo=timezone.utc),
        "rationale": (
            "Monthly Heparin Sodium replenishment at DC-Texas. Stable surgical volume with "
            "predictable anticoagulant demand at 55 units per day. CoagPharma Ltd selected for "
            "97% fill rate and reliable cold-chain handling. Standard 30-day coverage order."
        ),
    },
    {
        "sku": "MED-6201", "location": "DC-Texas",
        "supplier_id": "SUP-014", "supplier_name": "AnticoagDirect",
        "quantity": 300, "unit_price": 3.25, "total_cost": 975.00,
        "days_of_stock_at_order": 3.5, "trend_at_order": "increasing",
        "outcome": "delayed",
        "ordered_at": datetime(2024, 7, 3, 14, 0, tzinfo=timezone.utc),
        "rationale": (
            "Emergency supplemental Heparin order through AnticoagDirect when CoagPharma "
            "reported a temporary supply shortfall. Secondary supplier used despite lower fill "
            "rate and longer lead time. Order delayed 3 days — reinforced importance of earlier "
            "reorder triggers and maintaining a vetted secondary supplier relationship."
        ),
    },
    # DIAG-0331 historical orders
    {
        "sku": "DIAG-0331", "location": "DC-Ohio",
        "supplier_id": "SUP-015", "supplier_name": "QuickDiag Labs",
        "quantity": 2500, "unit_price": 4.50, "total_cost": 11250.00,
        "days_of_stock_at_order": 1.8, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 12, 18, 7, 0, tzinfo=timezone.utc),
        "rationale": (
            "COVID/Flu test kit stock near depletion at 1.8 days during peak winter respiratory "
            "season. Consumption tripled week-over-week due to a local influenza A outbreak. "
            "Emergency order of 2500 kits via QuickDiag Labs leveraging their surge reserve. "
            "Fastest lead time supplier selected given ED patient throughput dependency."
        ),
    },
    {
        "sku": "DIAG-0331", "location": "DC-Ohio",
        "supplier_id": "SUP-015", "supplier_name": "QuickDiag Labs",
        "quantity": 1500, "unit_price": 4.50, "total_cost": 6750.00,
        "days_of_stock_at_order": 4.5, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 10, 5, 9, 0, tzinfo=timezone.utc),
        "rationale": (
            "Pre-season respiratory test kit reorder at DC-Ohio ahead of expected winter flu "
            "surge. Stock at 4.5 days with stable summer consumption. QuickDiag Labs selected "
            "for 2-day lead time and strong FDA EUA documentation. Quantity sized for 30-day "
            "peak-season coverage including safety buffer."
        ),
    },
    {
        "sku": "DIAG-0331", "location": "DC-Ohio",
        "supplier_id": "SUP-016", "supplier_name": "TestBridge Inc",
        "quantity": 1000, "unit_price": 4.60, "total_cost": 4600.00,
        "days_of_stock_at_order": 3.2, "trend_at_order": "decreasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 7, 18, 13, 0, tzinfo=timezone.utc),
        "rationale": (
            "Summer supplemental test kit order via TestBridge Inc capitalising on bulk pricing "
            "during the low-demand inter-season period. Consumption declining post-spring. "
            "Cost savings at 1000-unit order offset the longer 5-day lead time for this "
            "non-urgent forward stockpile intended for autumn respiratory season readiness."
        ),
    },
    # SURG-2244 historical orders
    {
        "sku": "SURG-2244", "location": "DC-California",
        "supplier_id": "SUP-017", "supplier_name": "RespiraTech PPE",
        "quantity": 1500, "unit_price": 2.75, "total_cost": 4125.00,
        "days_of_stock_at_order": 1.5, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 11, 30, 6, 30, tzinfo=timezone.utc),
        "rationale": (
            "N95 respirator masks at critical 1.5 days of stock with consumption surging due "
            "to a respiratory illness cluster requiring enhanced airborne isolation precautions. "
            "Emergency order placed with RespiraTech PPE for their fastest domestic 3-day "
            "fulfillment. NIOSH-approved masks required for infection control protocol compliance."
        ),
    },
    {
        "sku": "SURG-2244", "location": "DC-California",
        "supplier_id": "SUP-017", "supplier_name": "RespiraTech PPE",
        "quantity": 800, "unit_price": 2.75, "total_cost": 2200.00,
        "days_of_stock_at_order": 5.0, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 9, 25, 10, 0, tzinfo=timezone.utc),
        "rationale": (
            "Monthly N95 mask replenishment at DC-California. Stable consumption at approximately "
            "70 units per day for routine surgical and isolation use. RespiraTech PPE preferred "
            "for 98.5% fill rate and NIOSH compliance. Ordered 800 units for 30-day coverage "
            "plus safety stock restoration."
        ),
    },
    {
        "sku": "SURG-2244", "location": "DC-California",
        "supplier_id": "SUP-018", "supplier_name": "N95Shield Supply",
        "quantity": 2000, "unit_price": 2.65, "total_cost": 5300.00,
        "days_of_stock_at_order": 7.5, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 6, 20, 14, 0, tzinfo=timezone.utc),
        "rationale": (
            "Strategic bulk N95 procurement via N95Shield Supply for pandemic preparedness "
            "stockpile during low-demand summer period. Adequate stock at 7.5 days. Chose "
            "N95Shield for best bulk unit pricing at 2000+ quantity. Extended 5-day lead "
            "time acceptable given the non-urgent strategic reserve nature of this order."
        ),
    },
    # LAB-0112 historical orders
    {
        "sku": "LAB-0112", "location": "DC-Texas",
        "supplier_id": "SUP-019", "supplier_name": "LabCore Supplies",
        "quantity": 500, "unit_price": 6.00, "total_cost": 3000.00,
        "days_of_stock_at_order": 2.2, "trend_at_order": "increasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 12, 5, 9, 0, tzinfo=timezone.utc),
        "rationale": (
            "Aerobic blood culture bottles nearly depleted at 2.2 days remaining with a spike "
            "in bacteremia workups from the ICU and ED. Consumption doubled due to a community-"
            "acquired sepsis cluster. Emergency order placed with LabCore Supplies for rapid "
            "3-day fulfillment to maintain diagnostic capability for sepsis workup programs."
        ),
    },
    {
        "sku": "LAB-0112", "location": "DC-Texas",
        "supplier_id": "SUP-019", "supplier_name": "LabCore Supplies",
        "quantity": 350, "unit_price": 6.00, "total_cost": 2100.00,
        "days_of_stock_at_order": 5.8, "trend_at_order": "stable",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 10, 14, 11, 0, tzinfo=timezone.utc),
        "rationale": (
            "Monthly blood culture bottle replenishment at DC-Texas. Stable laboratory workload "
            "at approximately 35 bottles per day. LabCore Supplies selected for 97.5% fill rate "
            "and ISO 13485 certification. Routine 30-day coverage order with safety stock buffer."
        ),
    },
    {
        "sku": "LAB-0112", "location": "DC-Texas",
        "supplier_id": "SUP-020", "supplier_name": "ClinPath Direct",
        "quantity": 200, "unit_price": 5.80, "total_cost": 1160.00,
        "days_of_stock_at_order": 4.0, "trend_at_order": "decreasing",
        "outcome": "delivered_on_time",
        "ordered_at": datetime(2024, 8, 9, 13, 30, tzinfo=timezone.utc),
        "rationale": (
            "Supplemental blood culture bottle order via ClinPath Direct for cost savings "
            "during a lower-demand period. Post-summer lab volume declining with consumption "
            "trending downward. ClinPath selected for best unit price with acceptable 6-day "
            "lead time given adequate stock buffer for this non-urgent top-up."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Historical proposed_orders — pre-seeds the procedure extractor so candidate
# rules are visible in the dashboard on day 1 of a demo.
#
# Two clusters (each ≥ _MIN_APPROVALS = 5 within the 30-day look-back):
#   pharmaceutical @ DC-Texas → BioPharm Global (SUP-003)  6 orders
#   surgical       @ DC-Ohio  → SafeGlove Co    (SUP-005)  6 orders
#
# Rules are left at human_confirmed=False so the confirmation step in the
# Streamlit dashboard is still visible during the demo.
# ---------------------------------------------------------------------------
_SEED_ORDERS: list[dict] = [
    # ── pharmaceutical @ DC-Texas → BioPharm Global (SUP-003) ──────────────
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 3600, "unit_price": 11.80, "total_cost": 42480.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": True,
     "rationale": "Monthly Insulin Glargine replenishment. BioPharm Global selected for cold-chain compliance and 98.5% fill rate.",
     "days_ago": 28},
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 3600, "unit_price": 11.80, "total_cost": 42480.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": True,
     "rationale": "Insulin Glargine reorder. Rising consumption trend; BioPharm Global preferred for biologics cold-chain.",
     "days_ago": 23},
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 3800, "unit_price": 11.80, "total_cost": 44840.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": False,
     "rationale": "Insulin stock at 3.1 days. BioPharm Global selected for GDP compliance and cold-chain capability.",
     "days_ago": 18},
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 3600, "unit_price": 11.80, "total_cost": 42480.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": True,
     "rationale": "Routine Insulin Glargine top-up. BioPharm Global preferred for 96% on-time delivery.",
     "days_ago": 14},
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 4000, "unit_price": 11.80, "total_cost": 47200.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": False,
     "rationale": "Insulin demand spike from new hospital contract. Emergency order BioPharm Global cold-chain delivery.",
     "days_ago": 9},
    {"sku": "MED-3017", "location": "DC-Texas",
     "supplier_id": "SUP-003", "supplier_name": "BioPharm Global",
     "quantity_recommended": 3600, "unit_price": 11.80, "total_cost": 42480.00,
     "expected_delivery_days": 10, "confidence": "high", "auto_approved": True,
     "rationale": "Scheduled Insulin Glargine replenishment. BioPharm Global — preferred pharmaceutical biologics supplier.",
     "days_ago": 4},

    # ── surgical @ DC-Ohio → SafeGlove Co (SUP-005) ─────────────────────────
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 540, "unit_price": 8.50, "total_cost": 4590.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": True,
     "rationale": "Nitrile gloves reorder. SafeGlove Co preferred for 99% fill rate and 3-day lead time.",
     "days_ago": 27},
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 600, "unit_price": 8.50, "total_cost": 5100.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": True,
     "rationale": "PPE surge reorder. SafeGlove Co selected for fastest lead time during elevated surgical volume.",
     "days_ago": 21},
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 540, "unit_price": 8.50, "total_cost": 4590.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": True,
     "rationale": "Routine glove replenishment. SafeGlove Co chosen for rapid fulfillment capability.",
     "days_ago": 16},
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 600, "unit_price": 8.50, "total_cost": 5100.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": False,
     "rationale": "Nitrile gloves critically low. SafeGlove Co emergency fulfillment for patient safety.",
     "days_ago": 11},
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 540, "unit_price": 8.50, "total_cost": 4590.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": True,
     "rationale": "Monthly glove stock restoration. SafeGlove Co preferred for FDA 510(k) compliance and fill rate.",
     "days_ago": 6},
    {"sku": "SURG-0084", "location": "DC-Ohio",
     "supplier_id": "SUP-005", "supplier_name": "SafeGlove Co",
     "quantity_recommended": 580, "unit_price": 8.50, "total_cost": 4930.00,
     "expected_delivery_days": 3, "confidence": "high", "auto_approved": True,
     "rationale": "Nitrile glove top-up before weekend surgical schedule. SafeGlove Co selected for reliability.",
     "days_ago": 2},

]


def _get_embedding_with_retry(text: str, max_retries: int = 3) -> list:
    """Embed a document string via the configured embedding provider
    (agent/embeddings.py). Retries with backoff are handled by the provider
    itself; max_retries is kept only for backward-compatible call sites.
    """
    return _embeddings.embed_documents([text])[0]


def _drop_and_recreate_ts(name: str, time_field: str, meta_field: str) -> None:
    if name in db.list_collection_names():
        db.drop_collection(name)
    db.create_collection(
        name,
        timeseries={"timeField": time_field, "metaField": meta_field},
    )
    print(f"  Created time series collection '{name}'")


def _wait_for_index(collection, index_name: str, timeout: int = 180) -> bool:
    """Poll until a Search/Vector Search index reaches READY status."""
    print(f"  Waiting for index '{index_name}' to become READY ", end="", flush=True)
    for _ in range(timeout // 5):
        try:
            indexes = list(collection.list_search_indexes(name=index_name))
            if indexes and indexes[0].get("status") == "READY":
                print(" ready.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timed out.")
    return False


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------

def seed_inventory() -> None:
    print("Seeding inventory …")
    db.inventory.drop()
    db.inventory.create_index([("sku", ASCENDING), ("location", ASCENDING)], unique=True)
    db.inventory.insert_many(SKUS)
    print(f"  Inserted {len(SKUS)} SKUs")


def seed_suppliers() -> None:
    print("Seeding suppliers (with notes for Atlas Search) …")
    db.suppliers.drop()
    db.suppliers.create_index([("sku", ASCENDING)])
    # fill_rate_pct index: get_supplier_options sorts by fill_rate_pct DESC
    db.suppliers.create_index([("fill_rate_pct", ASCENDING)])
    db.suppliers.insert_many(SUPPLIERS)
    print(f"  Inserted {len(SUPPLIERS)} supplier records")


def seed_consumption_history() -> None:
    """Seed 90 days of consumption history into the Time Series collection.

    Flat SKUs: Gaussian noise around a constant mean — trend will be "stable".

    Rising-trend SKUs (RISING_TREND_SKUS):
      • Days 1–(90-_RAMP_DAYS): flat at base average — establishes a stable baseline.
      • Days (90-_RAMP_DAYS+1)–90: mean scales linearly from 1.0× to 2.0× the base.

    With _RAMP_DAYS=30 the trend detector's 14-day window (7-day halves) sees:
      first 7 days  ≈ 1.64× base   (days 77-83 of the ramp)
      second 7 days ≈ 1.89× base   (days 84-90 of the ramp)
      ratio ≈ 1.15 > 1.10 threshold → reliably yields trend="increasing".
    """
    print("Seeding consumption_history (time series, 90 days) …")
    _drop_and_recreate_ts("consumption_history", "timestamp", "sku")

    docs = []
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    reasons = ["daily_dispensing", "emergency_issue", "scheduled_procedure", "routine_restock"]

    ramp_start_offset = _RAMP_DAYS  # day_offset at which ramp begins

    for day_offset in range(90, 0, -1):
        ts = now - timedelta(days=day_offset)
        for sku, avg in AVG_DAILY.items():
            if sku in RISING_TREND_SKUS and day_offset <= ramp_start_offset:
                # Linear ramp: oldest ramp day → scale=1.0, most recent → scale=2.0
                # day_offset counts down: ramp_start_offset=30 → 1
                scale = 1.0 + (ramp_start_offset - day_offset) / (ramp_start_offset - 1)
                mean_per_event = (avg / 2) * scale
            else:
                mean_per_event = avg / 2

            for hour in (9, 14):
                qty = max(1, int(random.gauss(mean_per_event, avg * 0.1)))
                docs.append({
                    "timestamp": ts.replace(hour=hour),
                    "sku": sku,
                    "location": LOCATION_BY_SKU[sku],
                    "quantity": qty,
                    "reason": random.choice(reasons),
                })

    db.consumption_history.insert_many(docs)
    rising = ", ".join(sorted(RISING_TREND_SKUS))
    print(f"  Inserted {len(docs)} consumption events")
    print(f"  Rising-trend SKUs (last {_RAMP_DAYS} days): {rising}")


def seed_order_history() -> None:
    """Seed historical orders WITH embeddings for Vector Search.

    Each order's rationale is embedded with Voyage AI (with retry). If
    embeddings fail for an individual order it is skipped with a warning
    rather than aborting the whole seeding run.  If ALL embeddings fail
    the orders are still inserted without embeddings so inventory / supplier
    data remain usable — Vector Search simply won't return results.
    """
    print("Seeding order_history with embeddings (Vector Search) …")
    db.order_history.drop()
    db.order_history.create_index([("sku", ASCENDING)])
    db.order_history.create_index([("proposed_order_id", ASCENDING)])

    docs = []
    failed = 0
    for i, order in enumerate(ORDER_HISTORY):
        print(f"  Embedding order {i+1}/{len(ORDER_HISTORY)}: {order['sku']} …", end="\r")
        # Inject category so the Vector Search pre-filter on order_history_vector_index
        # can scope ANN search to same-category orders at query time.
        enriched = {**order, "category": CATEGORY_BY_SKU.get(order["sku"], "unknown")}
        try:
            embedding = _get_embedding_with_retry(enriched["rationale"])
            docs.append({**enriched, "embedding": embedding})
        except Exception as exc:
            failed += 1
            print(f"\n  [WARN] Skipping embedding for {order['sku']}: {exc}")
            docs.append(enriched)  # insert without embedding — no Vector Search hit

    db.order_history.insert_many(docs)
    embedded = len(ORDER_HISTORY) - failed
    print(f"\n  Inserted {len(docs)} historical orders ({embedded} with embeddings, {failed} without)")


def seed_proposed_orders() -> None:
    """Insert historical approved proposed_orders to pre-seed the procedure extractor.

    Creates ≥5 approved orders per (supplier, category, location) cluster so
    seed_procedure_candidates() can immediately extract candidate rules without
    waiting for the demo to accumulate real orders.

    All dates fall within the 30-day look-back window of procedure_extractor.py.
    alert_id is a synthetic ObjectId — these records have no corresponding
    reorder_alert document (they represent pre-existing procurement history).
    """
    print("Seeding historical proposed_orders for procedure extraction …")
    now = datetime.now(timezone.utc)
    docs = []
    for order in _SEED_ORDERS:
        created_at = now - timedelta(days=order["days_ago"])
        docs.append({
            "sku":                    order["sku"],
            "location":               order["location"],
            "supplier_id":            order["supplier_id"],
            "supplier_name":          order["supplier_name"],
            "quantity_recommended":   order["quantity_recommended"],
            "unit_price":             order["unit_price"],
            "total_cost":             order["total_cost"],
            "expected_delivery_days": order["expected_delivery_days"],
            "rationale":              order["rationale"],
            "confidence":             order["confidence"],
            "similar_orders":         [],
            "atlas_search_used":      True,
            "retrieval_trace":        [],
            # "received" = already delivered.
            # The simulator's deliver_pending_orders() only queries "approved"
            # orders — using "received" here prevents it from re-delivering
            # these seeded orders and inflating on_hand at startup.
            # The procedure extractor queries status in ["approved", "received"]
            # so it still discovers these records.
            # assess_alert's existing_order_qty only counts ["awaiting_approval",
            # "approved"], so these records don't create a false coverage gap.
            "status":                 "received",
            "auto_approved":          order["auto_approved"],
            "review_reason":          None,
            "created_at":             created_at,
            "alert_id":               ObjectId(),   # synthetic — no real alert doc
        })
    db.proposed_orders.insert_many(docs)
    human_ct = sum(1 for o in _SEED_ORDERS if not o["auto_approved"])
    auto_ct  = len(_SEED_ORDERS) - human_ct
    print(f"  Inserted {len(docs)} historical approved orders "
          f"({auto_ct} auto-approved, {human_ct} human-approved)")


def seed_procedure_candidates() -> None:
    """Run the procedure extractor to derive candidate rules from seeded orders.

    Writes rules with human_confirmed=False so the confirmation step is still
    visible in the Streamlit dashboard during the demo.
    """
    print("Extracting procedure candidates from seeded proposed_orders …")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agent.procedure_extractor import extract_procedures
        rules = extract_procedures()
        if rules:
            print(f"  ✓ {len(rules)} candidate rule(s) written (human_confirmed=False)")
            for r in rules:
                print(f"    {r['sku_category']} @ {r['location']} → "
                      f"{r['preferred_supplier_name']} "
                      f"(evidence={r['evidence_count']})")
        else:
            print("  No new rules extracted (already exist or threshold not met)")
    except Exception as exc:
        print(f"  [WARN] Procedure extraction failed: {exc}")
        print("  Run 'python agent/procedure_extractor.py' manually after seeding.")


def seed_clear_alerts_and_orders() -> None:
    print("Clearing reorder_alerts, proposed_orders, short-term memory, and checkpoints …")
    print("  (long-term agent_memory is preserved across resets)")
    db.reorder_alerts.drop()
    db.reorder_alerts.create_index([("status", ASCENDING)])
    db.reorder_alerts.create_index([("sku", ASCENDING), ("created_at", ASCENDING)])
    # Compound (status, created_at): UI query find({"status":"pending"}).sort("created_at",-1)
    db.reorder_alerts.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    db.proposed_orders.drop()
    db.proposed_orders.create_index([("status", ASCENDING)])
    db.proposed_orders.create_index([("alert_id", ASCENDING)])
    # Compound (sku, location, status): assess_alert queries this on every alert
    db.proposed_orders.create_index([("sku", ASCENDING), ("location", ASCENDING), ("status", ASCENDING)])
    # created_at: UI sort find({}).sort("created_at", -1) fires every 5 s
    db.proposed_orders.create_index([("created_at", ASCENDING)])
    db.short_term_memory.drop()
    # Clear LangGraph checkpoints — these grow unboundedly during a demo session
    # (one checkpoint per graph node per alert). Dropping on reseed keeps the
    # collection from bloating across multiple demo runs.
    db.checkpoints.drop()
    db.checkpoint_writes.drop()
    # Clear the Change Stream resume token — reorder_alerts is being dropped so
    # the saved token is now invalid.  The agent will open a fresh stream on its
    # next reconnect rather than failing on a non-existent resume position.
    db.agent_state.delete_one({"_id": "change_stream_resume_token"})
    # Reset simulator control state so the demo starts in "running" mode
    db.simulator_control.update_one(
        {"_id": "main"}, {"$set": {"state": "running", "speed": 1}}, upsert=True
    )
    # Clear Phase-2/3/4 operational collections so demo starts clean
    for coll in (
        "alert_lifecycle",
        "confidence_outcomes",
        "escalation_queue",
        "failed_memory_writes",
        "human_review_queue",
        "dead_letter_events",
        "procedures",
    ):
        db[coll].drop()
    # Ensure lightweight indexes on commonly-queried fields
    db.alert_lifecycle.create_index([("sku", ASCENDING), ("location", ASCENDING)])
    db.confidence_outcomes.create_index([("alert_id", ASCENDING)])
    db.confidence_outcomes.create_index([("predicted_confidence", ASCENDING), ("outcome", ASCENDING)])
    db.escalation_queue.create_index([("sku", ASCENDING), ("location", ASCENDING)])
    db.failed_memory_writes.create_index([("resolved", ASCENDING), ("retry_count", ASCENDING)])
    # get_applicable_procedures queries (sku_category, location, human_confirmed)
    db.procedures.create_index(
        [("sku_category", ASCENDING), ("location", ASCENDING), ("human_confirmed", ASCENDING)]
    )
    print("  Done")


def seed_initial_alerts() -> None:
    """Create validated reorder_alerts for SKUs below reorder point at seed time."""
    print("Creating initial alerts for below-reorder-point SKUs …")
    inserted = 0
    now = datetime.now(timezone.utc)
    for sku_doc in SKUS:
        sku       = sku_doc["sku"]
        on_hand   = sku_doc["on_hand"]
        on_order  = sku_doc["on_order"]
        reorder   = sku_doc["reorder_point"]
        if on_hand + on_order >= reorder:
            continue
        avg_daily = AVG_DAILY.get(sku, 1)
        days_remaining = round(on_hand / avg_daily, 1) if avg_daily > 0 else 0.0
        alert = build_reorder_alert(
            sku=sku,
            location=sku_doc["location"],
            on_hand=on_hand,
            on_order=on_order,
            reorder_point=reorder,
            units_consumed_last_15min=0,
            avg_daily_consumption=avg_daily,
            days_of_stock_remaining=days_remaining,
            source="seed",
            created_at=now,
        )
        if insert_reorder_alert(db, alert, source="seed") is not None:
            inserted += 1
            print(f"  Alert → {sku} @ {sku_doc['location']}: "
                  f"on_hand={on_hand}, reorder_pt={reorder}, days_left={days_remaining}")
        else:
            print(f"  [WARN] Alert validation failed for {sku} @ {sku_doc['location']}")
    if inserted:
        print(f"  Inserted {inserted} initial alert(s)")
    else:
        print("  No SKUs below reorder point — no alerts created")


def create_atlas_indexes() -> None:
    """
    Create Atlas Search index (suppliers), Vector Search index (order_history),
    and Vector Search index (agent_memory for long-term semantic memory).
    Also creates a TTL index on short_term_memory for automatic 24 h expiry.
    These are Atlas-only features and require a connected Atlas cluster.
    Index creation is async — we poll until READY before returning.
    """
    # short_term_memory is dropped on reseed — recreate it before indexing.
    if "short_term_memory" not in db.list_collection_names():
        db.create_collection("short_term_memory")
        print("  Created empty 'short_term_memory' collection")

    # TTL index — MongoDB automatically deletes short-term memories after 24 h
    print("Creating TTL index on 'short_term_memory' …")
    db.short_term_memory.create_index("decided_at", expireAfterSeconds=86400)
    # Compound index: get_short_term_memories filters (sku, location) + sorts decided_at DESC
    db.short_term_memory.create_index(
        [("sku", ASCENDING), ("location", ASCENDING), ("decided_at", ASCENDING)]
    )
    print("  Done (TTL=86400 s)")

    print("\nCreating Atlas Search index on 'suppliers' …")
    try:
        existing = [i["name"] for i in db.suppliers.list_search_indexes()]
        if "suppliers_text_search" not in existing:
            db.suppliers.create_search_index({
                "name": "suppliers_text_search",
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "supplier_name": {"type": "string"},
                            "notes":         {"type": "string"},
                            "sku":           {"type": "string"},
                        },
                    }
                },
            })
            _wait_for_index(db.suppliers, "suppliers_text_search")
        else:
            print("  Index 'suppliers_text_search' already exists — skipping")
    except Exception as exc:
        print(f"  [WARN] Could not create Atlas Search index: {exc}")
        print("  Create it manually in the Atlas UI with fields: supplier_name, notes, sku (string)")

    print("Creating Atlas Vector Search index on 'order_history' …")
    try:
        existing = [i["name"] for i in db.order_history.list_search_indexes()]
        if "order_history_vector_index" in existing:
            print(f"  Dropping existing vector index to apply current dims={EMBEDDING_DIMS} …")
            db.order_history.drop_search_index("order_history_vector_index")
            # Poll until the index is fully removed before recreating
            for _ in range(60):
                still_there = [i["name"] for i in db.order_history.list_search_indexes()]
                if "order_history_vector_index" not in still_there:
                    break
                print(".", end="", flush=True)
                time.sleep(5)
            print()
        db.order_history.create_search_index({
            "name": "order_history_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type":          "vector",
                        "path":          "embedding",
                        "numDimensions": EMBEDDING_DIMS,
                        "similarity":    "cosine",
                    },
                    # Pre-filter field: allows $vectorSearch to scope results to
                    # same-category orders before ANN, improving relevance precision.
                    {
                        "type": "filter",
                        "path": "category",
                    },
                ]
            },
        })
        _wait_for_index(db.order_history, "order_history_vector_index")
    except Exception as exc:
        print(f"  [WARN] Could not create Vector Search index: {exc}")
        print(f"  Create it manually in the Atlas UI: field=embedding, dims={EMBEDDING_DIMS}, similarity=cosine")

    print("Creating Atlas Vector Search index on 'agent_memory' (long-term semantic memory) …")
    try:
        # Atlas requires the collection to exist before a search index can be created.
        # agent_memory starts empty and grows as the agent makes decisions, so we
        # create it explicitly here if it doesn't already exist.
        if "agent_memory" not in db.list_collection_names():
            db.create_collection("agent_memory")
            print("  Created empty 'agent_memory' collection")

        existing = [i["name"] for i in db.agent_memory.list_search_indexes()]
        if "agent_memory_vector_index" in existing:
            print("  Dropping existing agent_memory index to recreate …")
            db.agent_memory.drop_search_index("agent_memory_vector_index")
            for _ in range(60):
                still_there = [i["name"] for i in db.agent_memory.list_search_indexes()]
                if "agent_memory_vector_index" not in still_there:
                    break
                print(".", end="", flush=True)
                time.sleep(5)
            print()
        db.agent_memory.create_search_index({
            "name": "agent_memory_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type":          "vector",
                        "path":          "embedding",
                        "numDimensions": EMBEDDING_DIMS,
                        "similarity":    "cosine",
                    },
                    # Pre-filter field: allows $vectorSearch to scope memories to
                    # the same warehouse location when needed, without forcing it.
                    {
                        "type": "filter",
                        "path": "location",
                    },
                ]
            },
        })
        _wait_for_index(db.agent_memory, "agent_memory_vector_index")
    except Exception as exc:
        print(f"  [WARN] Could not create agent_memory Vector Search index: {exc}")
        print(f"  Create it manually in the Atlas UI: field=embedding, dims={EMBEDDING_DIMS}, similarity=cosine")


def _wait_for_mongo(max_attempts: int = 5, delay: int = 5) -> None:
    """Retry server_info() with backoff — Docker cold-start can be slow."""
    for attempt in range(1, max_attempts + 1):
        try:
            version = client.server_info()["version"]
            print(f"Connected to MongoDB {version}\n")
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise
            print(
                f"[{attempt}/{max_attempts}] MongoDB not reachable yet ({exc}). "
                f"Retrying in {delay}s …"
            )
            time.sleep(delay)
            delay = min(delay * 2, 30)


if __name__ == "__main__":
    _wait_for_mongo()
    seed_inventory()
    seed_suppliers()
    seed_consumption_history()
    try:
        seed_order_history()
    except Exception as exc:
        print(f"\n[ERROR] seed_order_history failed: {exc}")
        print("  Inventory and supplier data are intact. Vector Search will be unavailable.")
        print("  Fix the Voyage AI credentials and re-run seed.py to add embeddings.")
    seed_clear_alerts_and_orders()
    seed_proposed_orders()        # historical approved orders → feeds procedure extractor
    create_atlas_indexes()
    seed_initial_alerts()
    seed_procedure_candidates()   # derive candidate rules → visible in dashboard
    print("\nSeed complete.")
    print("Run 'python simulator/stream_simulator.py' and 'python agent/graph.py' to start the demo.")
