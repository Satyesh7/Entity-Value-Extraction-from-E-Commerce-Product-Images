import pandas as pd
import cv2
import numpy as np
import re
import requests
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Set use_gpu=False if you don't have a GPU

entity_unit_map = {
    'width': {'centimetre', 'centimeter', 'cm', 'Cm', 'CM', 'foot', 'feet', 'ft', 'Ft', 'FT', 'inch', 'in', 'In', 'IN', 'metre', 'meter', 'm', 'M', 'millimetre', 'millimeter', 'mm', 'Mm', 'MM', 'yard', 'yd', 'Yd', 'YD'},
    'depth': {'centimetre', 'centimeter', 'cm', 'Cm', 'CM', 'foot', 'feet', 'ft', 'Ft', 'FT', 'inch', 'in', 'In', 'IN', 'metre', 'meter', 'm', 'M', 'millimetre', 'millimeter', 'mm', 'Mm', 'MM', 'yard', 'yd', 'Yd', 'YD'},
    'height': {'centimetre', 'centimeter', 'cm', 'Cm', 'CM', 'foot', 'feet', 'ft', 'Ft', 'FT', 'inch', 'in', 'In', 'IN', 'metre', 'meter', 'm', 'M', 'millimetre', 'millimeter', 'mm', 'Mm', 'MM', 'yard', 'yd', 'Yd', 'YD'},
    'item_weight': {'gram', 'Gram', 'GRAM', 'g', 'G', 'gm', 'Gm', 'GM', 'grm', 'Grm', 'GRM', 'kilogram', 'Kilogram', 'KiloGram', 'KILOGRAM', 'kg', 'Kg', 'KG', 'kG', 'kilo', 'Kilo', 'KILO', 'microgram', 'Microgram', 'MICROGRAM', 'μg', 'ug', 'Ug', 'UG', 'milligram', 'Milligram', 'MILLIGRAM', 'mg', 'Mg', 'MG', 'mG', 'ounce', 'Ounce', 'OUNCE', 'oz', 'Oz', 'OZ', 'pound', 'Pound', 'POUND', 'lb', 'Lb', 'LB', 'lB', 'lbs', 'Lbs', 'LBS', 'ton', 'Ton', 'TON', 't', 'T'},
    'maximum_weight_recommendation': {'gram', 'Gram', 'GRAM', 'g', 'G', 'gm', 'Gm', 'GM', 'grm', 'Grm', 'GRM', 'kilogram', 'Kilogram', 'KiloGram', 'KILOGRAM', 'kg', 'Kg', 'KG', 'kG', 'kilo', 'Kilo', 'KILO', 'microgram', 'Microgram', 'MICROGRAM', 'μg', 'ug', 'Ug', 'UG', 'milligram', 'Milligram', 'MILLIGRAM', 'mg', 'Mg', 'MG', 'mG', 'ounce', 'Ounce', 'OUNCE', 'oz', 'Oz', 'OZ', 'pound', 'Pound', 'POUND', 'lb', 'Lb', 'LB', 'lB', 'lbs', 'Lbs', 'LBS', 'ton', 'Ton', 'TON', 't', 'T'},
    'voltage': {'kilovolt', 'Kilovolt', 'KILOVOLT', 'kV', 'KV', 'kv', 'Kv', 'millivolt', 'Millivolt', 'MILLIVOLT', 'mV', 'MV', 'mv', 'Mv', 'volt', 'Volt', 'VOLT', 'v', 'V'},
    'wattage': {'kilowatt', 'Kilowatt', 'KILOWATT', 'kW', 'KW', 'kw', 'Kw', 'watt', 'Watt', 'WATT', 'w', 'W'},
    'item_volume': {'centilitre', 'Centilitre', 'CENTILITRE', 'centiliter', 'Centiliter', 'CENTILITER', 'cl', 'Cl', 'CL', 'cubic foot', 'Cubic Foot', 'CUBIC FOOT', 'cu ft', 'Cu Ft', 'CU FT', 'cubic inch', 'Cubic Inch', 'CUBIC INCH', 'cu in', 'Cu In', 'CU IN', 'cup', 'Cup', 'CUP', 'c', 'C', 'decilitre', 'Decilitre', 'DECILITRE', 'deciliter', 'Deciliter', 'DECILITER', 'dl', 'Dl', 'DL', 'fluid ounce', 'Fluid Ounce', 'FLUID OUNCE', 'fl oz', 'Fl Oz', 'FL OZ', 'gallon', 'Gallon', 'GALLON', 'gal', 'Gal', 'GAL', 'imperial gallon', 'Imperial Gallon', 'IMPERIAL GALLON', 'imp gal', 'Imp Gal', 'IMP GAL', 'litre', 'Litre', 'LITRE', 'liter', 'Liter', 'LITER', 'l', 'L', 'microlitre', 'Microlitre', 'MICROLITRE', 'microliter', 'Microliter', 'MICROLITER', 'μl', 'ul', 'Ul', 'UL', 'millilitre', 'Millilitre', 'MILLILITRE', 'milliliter', 'Milliliter', 'MILLILITER', 'ml', 'Ml', 'ML', 'mL', 'pint', 'Pint', 'PINT', 'pt', 'Pt', 'PT', 'quart', 'Quart', 'QUART', 'qt', 'Qt', 'QT'}
}

def extract_entity_value(text, entity_name):
    allowed_units = entity_unit_map.get(entity_name, set())
    
    # Create a pattern that matches any number followed by any unit or its variations
    units_pattern = '|'.join(map(re.escape, allowed_units))
    pattern = r'(\d+(?:,\d+)(?:\.\d+)?)\s({})'.format(units_pattern)
    
    matches = re.findall(pattern, text)
    
    if matches:
        # Sort matches by numeric value (descending) and return the largest
        sorted_matches = sorted(matches, key=lambda x: float(x[0].replace(',', '')), reverse=True)
        value, unit = sorted_matches[0]
        
        # Normalize the unit to a standard form (lowercase)
        normalized_unit = unit.lower()
        
        return f"{float(value.replace(',', '')):.2f} {normalized_unit}"

    # If no match found with the above pattern, try a more lenient approach
    general_pattern = r'(\d+(?:,\d+)(?:\.\d+)?)\s(\w+)'
    matches = re.findall(general_pattern, text)
    for value, unit in matches:
        lower_unit = unit.lower()
        if any(u.lower().startswith(lower_unit) for u in allowed_units):
            normalized_unit = next((u.lower() for u in allowed_units if u.lower().startswith(lower_unit)), lower_unit)
            return f"{float(value.replace(',', '')):.2f} {normalized_unit}"

    # If still no match, look for specific keywords (e.g., for weights)
    if entity_name in ['item_weight', 'maximum_weight_recommendation']:
        weight_patterns = [
            r'net\s*wt\.?\s*(\d+(?:,\d+)(?:\.\d+)?)\s(\w+)',
            r'weight:?\s*(\d+(?:,\d+)(?:\.\d+)?)\s(\w+)',
            r'(\d+(?:,\d+)(?:\.\d+)?)\s(\w+)\s*net'
        ]
        for pattern in weight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value, unit = match.groups()
                lower_unit = unit.lower()
                if any(u.lower().startswith(lower_unit) for u in allowed_units):
                    normalized_unit = next((u.lower() for u in allowed_units if u.lower().startswith(lower_unit)), lower_unit)
                    return f"{float(value.replace(',', '')):.2f} {normalized_unit}"

    return ""

def process_images(csv_file, num_images=100000):
    df = pd.read_csv(csv_file)
    df = df.head(num_images)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_link = row['image_link']
        entity_name = row['entity_name']
        
        try:
            response = requests.get(image_link)
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            
            ocr_result = ocr.ocr(image, cls=True)
            
            extracted_text = " ".join([line[1][0] for line in ocr_result[0]])
            prediction = extract_entity_value(extracted_text, entity_name)
            
        except Exception as e:
            prediction = ""
        
        results.append({
            'index': row.name,
            'prediction': prediction,
            'ground_truth': row['entity_value']
        })
    
    output_df = pd.DataFrame(results)
    return output_df

def normalize_value(value):
    if not value:
        return ""
    # Remove any commas and convert to lowercase
    value = value.replace(',', '').lower()
    # Split into number and unit
    parts = value.split()
    if len(parts) != 2:
        return value  # Return as is if it doesn't have the expected format
    number, unit = parts
    # Convert number to float and round to 3 decimal places
    try:
        number = round(float(number), 3)
    except ValueError:
        return value  # Return as is if number conversion fails
    
    # Comprehensive unit normalization map
    unit_map = {
        # Length units
        'centimetre': 'centimetre', 'centimeter': 'centimetre', 'cm': 'centimetre',
        'foot': 'foot', 'feet': 'foot', 'ft': 'foot',
        'inch': 'inch', 'in': 'inch',
        'metre': 'metre', 'meter': 'metre', 'm': 'metre',
        'millimetre': 'millimetre', 'millimeter': 'millimetre', 'mm': 'millimetre',
        'yard': 'yard', 'yd': 'yard',
        
        # Weight units
        'gram': 'gram', 'g': 'gram', 'gm': 'gram', 'grm': 'gram',
        'kilogram': 'kilogram', 'kg': 'kilogram', 'kilo': 'kilogram',
        'microgram': 'microgram', 'μg': 'microgram', 'ug': 'microgram',
        'milligram': 'milligram', 'mg': 'milligram',
        'ounce': 'ounce', 'oz': 'ounce',
        'pound': 'pound', 'lb': 'pound', 'lbs': 'pound',
        'ton': 'ton', 't': 'ton',
        
        # Voltage units
        'kilovolt': 'kilovolt', 'kv': 'kilovolt',
        'millivolt': 'millivolt', 'mv': 'millivolt',
        'volt': 'volt', 'v': 'volt',
        
        # Wattage units
        'kilowatt': 'kilowatt', 'kw': 'kilowatt',
        'watt': 'watt', 'w': 'watt',
        
        # Volume units
        'centilitre': 'centilitre', 'centiliter': 'centilitre', 'cl': 'centilitre',
        'cubic foot': 'cubic foot', 'cu ft': 'cubic foot',
        'cubic inch': 'cubic inch', 'cu in': 'cubic inch',
        'cup': 'cup', 'c': 'cup',
        'decilitre': 'decilitre', 'deciliter': 'decilitre', 'dl': 'decilitre',
        'fluid ounce': 'fluid ounce', 'fl oz': 'fluid ounce',
        'gallon': 'gallon', 'gal': 'gallon',
        'imperial gallon': 'imperial gallon', 'imp gal': 'imperial gallon',
        'litre': 'litre', 'liter': 'litre', 'l': 'litre',
        'microlitre': 'microlitre', 'microliter': 'microlitre', 'μl': 'microlitre', 'ul': 'microlitre',
        'millilitre': 'millilitre', 'milliliter': 'millilitre', 'ml': 'millilitre',
        'pint': 'pint', 'pt': 'pint',
        'quart': 'quart', 'qt': 'quart'
    }
    
    # Normalize the unit
    normalized_unit = unit_map.get(unit, unit)
    
    return f"{number} {normalized_unit}"

def custom_evaluate_predictions(predictions, ground_truth):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for pred, gt in zip(predictions, ground_truth):
        norm_pred = normalize_value(pred)
        norm_gt = normalize_value(gt)
        
        if norm_pred != "" and norm_gt != "" and norm_pred == norm_gt:
            true_positives += 1
        elif norm_pred != "" and norm_gt != "" and norm_pred != norm_gt:
            false_positives += 1
        elif norm_pred != "" and norm_gt == "":
            false_positives += 1
        elif norm_pred == "" and norm_gt != "":
            false_negatives += 1
        elif norm_pred == "" and norm_gt == "":
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall, {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

# Main execution
if __name__ == "__main__":
    input_csv = "/content/train.csv"
    output_csv = "final_f1score_using_paddleocr_images.csv"
    
    print("Starting image processing...")
    output_df = process_images(input_csv, num_images=100000)
    
    print("Calculating F1 score...")
    f1, precision, recall, metrics = custom_evaluate_predictions(output_df['prediction'], output_df['ground_truth'])
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Metrics: {metrics}")
    
    print("Saving predictions...")
    output_df[['index', 'prediction']].to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")
