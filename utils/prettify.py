"""
Prettify
"""

import unittest

def b(value) -> str|list[str]:
    """
    Convert a boolean or sequence of values to check and cross emoji.
    Accepts bool, list/tuple/array of bool or truthy/falsy values.
    """
    if isinstance(value, bool):
        return "✅" if value else "❌"
    elif isinstance(value, (list, tuple)):
        return ", ".join(b(bool(v)) for v in value)
    else:
        # Try to convert to bool
        try:
            return "✅" if bool(value) else "❌"
        except Exception:
            raise ValueError("value must be a boolean or a sequence of values convertible to bool")
        
## --- TESTS AND DEMO ---
class TestSuite(unittest.TestCase):
    def test_b(self):
        """
        Test that b function works correctly.
        """
        self.assertEqual(b(True), "✅")
        self.assertEqual(b(False), "❌")
        self.assertEqual(b([True, False]), "✅, ❌")
        self.assertEqual(b([True, True]), "✅, ✅")
        self.assertEqual(b([False, False]), "❌, ❌")
        self.assertEqual(b([True, False, True]), "✅, ❌, ✅")