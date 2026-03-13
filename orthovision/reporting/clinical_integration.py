import json
from datetime import datetime

class ClinicalReporting:
    """
    Generates DICOM-SR (Structured Report) metadata and FHIR-compliant clinical resources.
    """

    @staticmethod
    def generate_fhir_resource(patient_id: str,
                               measurements: dict,
                               findings: str) -> str:
        """
        Creates a HL7 FHIR Observation resource for integration with EMR systems.
        """
        resource = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging"
                }]
            }],
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "valueString": findings,
            "component": []
        }

        for key, val in measurements.items():
            resource["component"].append({
                "code": {"text": key},
                "valueString": str(val)
            })

        return json.dumps(resource, indent=2)

    @staticmethod
    def generate_dicom_sr_metadata(analysis_results: dict):
        """
        Skeleton for creating DICOM Structured Reporting tags (TID 1500 etc).
        """
        # In a real system, this would use pydicom to build a Nested Sequence
        sr_data = {
            "ContentSequence": [
                {"RelationshipType": "CONTAINS", "ValueType": "TEXT", "TextValue": analysis_results.get("diagnosis")},
                {"RelationshipType": "CONTAINS", "ValueType": "NUM", "NumericValue": analysis_results.get("measurement")}
            ]
        }
        return sr_data
