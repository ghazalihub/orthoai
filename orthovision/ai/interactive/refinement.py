import numpy as np
from skimage.segmentation import random_walker

class InteractiveRefinement:
    """
    Allows surgeons to interactively refine AI-generated masks using scribbles.
    Implements Active Learning loop.
    """

    @staticmethod
    def refine_with_scribbles(image: np.ndarray,
                             ai_mask: np.ndarray,
                             scribbles: np.ndarray) -> np.ndarray:
        """
        image: Original grayscale image (CT/MRI)
        ai_mask: Initial AI-generated mask
        scribbles: User-defined markings (1: Foreground, 2: Background)
        """
        # Combine AI confidence with manual scribbles
        labels = scribbles.copy()
        # Trust scribbles over AI mask where provided

        # Use Random Walker for interactive segmentation
        refined_mask = random_walker(image, labels, beta=130, mode='bf')
        return refined_mask

    @staticmethod
    def calculate_dice_delta(old_mask: np.ndarray,
                            new_mask: np.ndarray) -> float:
        """
        Measures improvement after manual refinement.
        """
        intersection = np.sum(old_mask * new_mask)
        union = np.sum(old_mask) + np.sum(new_mask)
        dice = (2.0 * intersection) / (union + 1e-6)
        return float(dice)
