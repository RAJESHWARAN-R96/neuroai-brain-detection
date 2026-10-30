"""
Database models for storing Patient Scans, Diagnostic History, and Metaheuristic Optimization Experiments.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PatientScan(db.Model):
    __tablename__ = 'patient_scans'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(64), nullable=False, index=True)
    patient_name = db.Column(db.String(128), default="Anonymous Patient")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Image file paths (relative to uploads directory)
    original_image = db.Column(db.String(256), nullable=False)
    preprocessed_image = db.Column(db.String(256), nullable=False)
    gradcam_image = db.Column(db.String(256), nullable=False)
    report_pdf = db.Column(db.String(256), nullable=True)
    
    # Prediction results
    predicted_class = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_tumor = db.Column(db.Boolean, nullable=False)
    model_used = db.Column(db.String(64), nullable=False)
    
    # Probability distribution
    prob_glioma = db.Column(db.Float, default=0.0)
    prob_meningioma = db.Column(db.Float, default=0.0)
    prob_notumor = db.Column(db.Float, default=0.0)
    prob_pituitary = db.Column(db.Float, default=0.0)
    
    # Clinical metrics
    tumor_area_percentage = db.Column(db.Float, default=0.0)
    anatomical_quadrant = db.Column(db.String(64), default="N/A")
    severity_index = db.Column(db.String(64), default="Normal")
    notes = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'original_image': f"/uploads/{self.original_image}",
            'preprocessed_image': f"/uploads/{self.preprocessed_image}",
            'gradcam_image': f"/uploads/{self.gradcam_image}",
            'report_pdf': f"/uploads/{self.report_pdf}" if self.report_pdf else None,
            'predicted_class': self.predicted_class,
            'confidence': round(self.confidence, 1),
            'is_tumor': self.is_tumor,
            'model_used': self.model_used,
            'probabilities': {
                'glioma': round(self.prob_glioma, 1),
                'meningioma': round(self.prob_meningioma, 1),
                'notumor': round(self.prob_notumor, 1),
                'pituitary': round(self.prob_pituitary, 1)
            },
            'tumor_area_percentage': round(self.tumor_area_percentage, 2),
            'anatomical_quadrant': self.anatomical_quadrant,
            'severity_index': self.severity_index,
            'notes': self.notes
        }


class OptimizationExperiment(db.Model):
    __tablename__ = 'optimization_experiments'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    algorithm = db.Column(db.String(32), nullable=False)
    task_type = db.Column(db.String(64), nullable=False) # 'feature_selection', 'ensemble_weights', 'hpo'
    pop_size = db.Column(db.Integer, default=20)
    max_iter = db.Column(db.Integer, default=25)
    best_fitness = db.Column(db.Float, nullable=False)
    execution_time_sec = db.Column(db.Float, nullable=False)
    
    # JSON strings
    convergence_data = db.Column(db.Text, nullable=False)
    result_metrics = db.Column(db.Text, nullable=False)

    def to_dict(self):
        import json
        return {
            'id': self.id,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': self.algorithm,
            'task_type': self.task_type,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'best_fitness': round(self.best_fitness, 4),
            'execution_time_sec': round(self.execution_time_sec, 2),
            'convergence_data': json.loads(self.convergence_data) if self.convergence_data else [],
            'result_metrics': json.loads(self.result_metrics) if self.result_metrics else {}
        }
