"""Initial schema for appointments, predictions, model_versions, doctors

Revision ID: 4bec3792c2fb
Revises: 
Create Date: 2025-07-08 16:18:47.073003

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4bec3792c2fb'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('appointments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('doctor_id', sa.String(), nullable=False),
    sa.Column('summary', sa.String(), nullable=True),
    sa.Column('cleaned_summary', sa.String(), nullable=True),
    sa.Column('outcome_label', sa.String(), nullable=True),
    sa.Column('outcome_encoded', sa.Integer(), nullable=True),
    sa.Column('length', sa.Integer(), nullable=True),
    sa.Column('word_count', sa.Integer(), nullable=True),
    sa.Column('avg_word_length', sa.Float(), nullable=True),
    sa.Column('sentence_count', sa.Integer(), nullable=True),
    sa.Column('has_diabetes', sa.Boolean(), nullable=True),
    sa.Column('has_hypertension', sa.Boolean(), nullable=True),
    sa.Column('has_asthma', sa.Boolean(), nullable=True),
    sa.Column('has_eczema', sa.Boolean(), nullable=True),
    sa.Column('has_migraines', sa.Boolean(), nullable=True),
    sa.Column('has_anemia', sa.Boolean(), nullable=True),
    sa.Column('has_headache', sa.Boolean(), nullable=True),
    sa.Column('has_abdominal_pain', sa.Boolean(), nullable=True),
    sa.Column('has_back_pain', sa.Boolean(), nullable=True),
    sa.Column('has_fatigue', sa.Boolean(), nullable=True),
    sa.Column('has_dizziness', sa.Boolean(), nullable=True),
    sa.Column('has_shortness_of_breath', sa.Boolean(), nullable=True),
    sa.Column('has_cough', sa.Boolean(), nullable=True),
    sa.Column('has_rash', sa.Boolean(), nullable=True),
    sa.Column('has_xray', sa.Boolean(), nullable=True),
    sa.Column('has_ct_scan', sa.Boolean(), nullable=True),
    sa.Column('has_mri', sa.Boolean(), nullable=True),
    sa.Column('has_blood_test', sa.Boolean(), nullable=True),
    sa.Column('has_ecg', sa.Boolean(), nullable=True),
    sa.Column('has_amoxicillin', sa.Boolean(), nullable=True),
    sa.Column('has_ibuprofen', sa.Boolean(), nullable=True),
    sa.Column('has_paracetamol', sa.Boolean(), nullable=True),
    sa.Column('has_lisinopril', sa.Boolean(), nullable=True),
    sa.Column('has_metformin', sa.Boolean(), nullable=True),
    sa.Column('has_ventolin', sa.Boolean(), nullable=True),
    sa.Column('has_prescription', sa.Boolean(), nullable=True),
    sa.Column('has_referral', sa.Boolean(), nullable=True),
    sa.Column('has_lifestyle', sa.Boolean(), nullable=True),
    sa.Column('has_dietary', sa.Boolean(), nullable=True),
    sa.Column('has_exercise', sa.Boolean(), nullable=True),
    sa.Column('referral_cardiology', sa.Boolean(), nullable=True),
    sa.Column('referral_neurology', sa.Boolean(), nullable=True),
    sa.Column('referral_orthopedics', sa.Boolean(), nullable=True),
    sa.Column('referral_dermatology', sa.Boolean(), nullable=True),
    sa.Column('is_initial_assessment', sa.Boolean(), nullable=True),
    sa.Column('is_follow_up', sa.Boolean(), nullable=True),
    sa.Column('is_test_result_discussion', sa.Boolean(), nullable=True),
    sa.Column('has_time_reference', sa.Boolean(), nullable=True),
    sa.Column('has_followup_scheduling', sa.Boolean(), nullable=True),
    sa.Column('medical_term_density', sa.Float(), nullable=True),
    sa.Column('text_complexity', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('doctors',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=True),
    sa.Column('cases', sa.Integer(), nullable=True),
    sa.Column('avg_score', sa.Float(), nullable=True),
    sa.Column('weighted_score', sa.Float(), nullable=True),
    sa.Column('outlier', sa.Boolean(), nullable=True),
    sa.Column('outlier_type', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('model_versions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('version', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('predictions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('appointment_id', sa.UUID(), nullable=True),
    sa.Column('model_version_id', sa.UUID(), nullable=True),
    sa.Column('prediction_label', sa.String(), nullable=True),
    sa.Column('prediction_score', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['appointment_id'], ['appointments.id'], ),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('predictions')
    op.drop_table('model_versions')
    op.drop_table('doctors')
    op.drop_table('appointments')
    # ### end Alembic commands ###
