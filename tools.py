"""
Tool implementations for the RAG-Powered Multi-Agent Q&A Assistant.

This module contains specialized tools that the agent can use for specific tasks.
"""

import re
import numpy as np
import requests
from urllib.parse import quote_plus


class Tools:
    """Collection of tools that the agent can use."""
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Sanitize the expression - only allow digits, operators, and some math functions
            safe_chars = set("0123456789+-*/().^ ")
            safe_funcs = {"sin", "cos", "tan", "sqrt", "log", "exp"}
            
            tokens = re.findall(r'[a-zA-Z]+|[0-9]+|\S', expression)
            for token in tokens:
                if token.isalpha() and token not in safe_funcs:
                    return f"Error: Function '{token}' not allowed for security reasons."
                
                if not token.isalpha() and not all(c in safe_chars for c in token):
                    return f"Error: Character in '{token}' not allowed for security reasons."
            
            # Replace ^ with **
            expression = expression.replace('^', '**')
            
            # Use a restricted environment
            locals_dict = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "log": np.log,
                "exp": np.exp
            }
            
            result = eval(expression, {"__builtins__": {}}, locals_dict)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    @staticmethod
    def define(term: str) -> str:
        """Get definition of a term using a dictionary API."""
        try:
            # Using Free Dictionary API
            term = quote_plus(term.strip())
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}")
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    entry = data[0]
                    definition = entry['meanings'][0]['definitions'][0]['definition']
                    part_of_speech = entry['meanings'][0]['partOfSpeech']
                    return f"Definition of '{term}' ({part_of_speech}): {definition}"
                return f"Found entry for '{term}' but couldn't extract definition."
            else:
                return f"No definition found for '{term}'."
        except Exception as e:
            return f"Error looking up definition: {str(e)}"