import pydicom
import SimpleITK as sitk
import os
import numpy as np
from typing import List, Optional, Dict, Any

class DICOMEngine:
    """
    Engine for loading, parsing, and anonymizing DICOM studies.
    """

    @staticmethod
    def load_series(directory: str) -> sitk.Image:
        """
        Load a DICOM series from a directory as a SimpleITK image.
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image

    @staticmethod
    def get_metadata(filepath: str) -> Dict[str, Any]:
        """
        Extract key metadata from a DICOM file.
        """
        ds = pydicom.dcmread(filepath)
        metadata = {
            "PatientID": getattr(ds, "PatientID", "Unknown"),
            "PatientName": str(getattr(ds, "PatientName", "Unknown")),
            "Modality": getattr(ds, "Modality", "Unknown"),
            "StudyDate": getattr(ds, "StudyDate", "Unknown"),
            "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown"),
            "PixelSpacing": getattr(ds, "PixelSpacing", None),
            "SliceThickness": getattr(ds, "SliceThickness", None),
        }
        return metadata

    @staticmethod
    def anonymize(filepath: str, output_path: str):
        """
        Anonymize a DICOM file by removing patient identifiers.
        """
        ds = pydicom.dcmread(filepath)
        ds.PatientName = "ANONYMOUS"
        ds.PatientID = "000000"
        ds.PatientBirthDate = ""
        ds.save_as(output_path)

    @staticmethod
    def to_nifti(image: sitk.Image, output_path: str):
        """
        Convert SimpleITK image to NIfTI format.
        """
        sitk.WriteImage(image, output_path)
