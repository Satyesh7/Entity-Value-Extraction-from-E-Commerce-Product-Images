# Entity-Value-Extraction-from-E-Commerce-Product-Images
Feature Extraction from Images:  the goal is to create a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital. 

As digital marketplaces expand, many products lack detailed textual descriptions, making it essential to obtain key details directly from images. These images provide important information such as weight, volume, voltage, wattage, dimensions, and many more, which are critical for digital stores.

Data Description:
The dataset consists of the following columns:

index: A unique identifier (ID) for the data sample.
image_link: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg  To download images, use the download_images function from src/utils.py. See sample code in src/test.ipynb.
group_id: Category code of the product.
entity_name: Product entity name. For example, “item_weight”.
entity_value: Product entity value. For example, “34 gram”.
Note: For test.csv, you will not see the column entity_value as it is the target variable.

Output Format:
The output file should be a CSV with 2 columns:

index: The unique identifier (ID) of the data sample. Note that the index should match the test record index.
prediction: A string which should have the following format: “x unit” where x is a float number in standard formatting and unit is one of the allowed units (allowed units are mentioned in the Appendix). The two values should be concatenated and have a space between them.
For example: “2 gram”, “12.5 centimetre”, “2.56 ounce” are valid.
Invalid cases: “2 gms”, “60 ounce/1.7 kilogram”, “2.2e2 kilogram”, etc.
Note: Make sure to output a prediction for all indices. If no value is found in the image for any test sample, return an empty string, i.e., “”. If you have less/more number of output samples in the output file as compared to test.csv, your output won’t be evaluated.

Evaluation Criteria:
Submissions will be evaluated based on the F1 score, which is a standard measure of prediction accuracy for classification and extraction problems.

Let GT = Ground truth value for a sample and OUT be the output prediction from the model for a sample. Then we classify the predictions into one of the 4 classes with the following logic:

True Positives: If OUT != "" and GT != "" and OUT == GT
False Positives: If OUT != "" and GT != "" and OUT != GT
False Positives: If OUT != "" and GT == ""
False Negatives: If OUT == "" and GT != ""
True Negatives: If OUT == "" and GT == ""
Then,
F1 score = 2 * Precision * Recall / (Precision + Recall)
where:

Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)


# Demo on one image:

Extracted Text:
HERBALIFE
BeverageMix
Regumenents
Bebida en Polvo
Protein-based snack for energy and nutrition
Bocadillo de proteina para energia y nutricion
enay
caa un
wild berry
moras
silvestres
naturally flavored
sabores naturales
15g
NET WT | PESO NETO: 9.88 OZ (280g
PROTEIN
PROTEINA


Entity_type: Weight
Entity_Value: 9.88 OZ
Extracted_Value: 9.88 OZ


![image](https://github.com/user-attachments/assets/0475182f-c812-4dcb-8b93-c3a24a1bf9be)

