"""Shared configuration for the private athlete analysis prototype."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "synthetic_athlete_data.csv"

PREDICTORS = [
    "muscular_strength",
    "lower_body_dynamics",
    "muscle_power_genetics",
    "blood_micronutrients",
    "basic_cognitive_function",
    "mental_health",
    "social_support",
    "training_conditions",
]

DOMAIN_ORDER = list(PREDICTORS)
DOMAIN_LABELS = {
    "muscular_strength": "Muscular\nStrength",
    "lower_body_dynamics": "Lower-Body\nDynamics",
    "muscle_power_genetics": "Muscle-Power\nGenetics",
    "blood_micronutrients": "Blood\nMicronutrients",
    "basic_cognitive_function": "Basic Cognitive\nFunction",
    "mental_health": "Mental\nHealth",
    "social_support": "Social\nSupport",
    "training_conditions": "Training\nConditions",
}

DISPLAY_NAMES = {
    "muscular_strength": "Muscular strength",
    "lower_body_dynamics": "Lower-body dynamics",
    "muscle_power_genetics": "Muscle-power genetics",
    "blood_micronutrients": "Blood micronutrients",
    "basic_cognitive_function": "Basic cognitive function",
    "mental_health": "Mental health",
    "social_support": "Social support",
    "training_conditions": "Training conditions",
    "age": "Age",
    "sex_female": "Female",
    "expertise_value": "Expertise value",
    "elite_status": "Elite status",
}

SPORTS = [
    "3x3 basketball",
    "ice hockey",
    "volleyball",
    "artistic gymnastics",
    "trampoline gymnastics",
    "rhythmic gymnastics",
    "table tennis",
    "modern pentathlon",
]
