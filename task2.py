import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    drug_path = "data/drug200.csv"
    # Loading dataset
    drug_data = pd.read_csv(drug_path)
