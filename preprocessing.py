import prettytable
from prettytable import PrettyTable
import pandas as pd
import copy
import numpy as np
import sklearn
class Pretty:
    def gen_table(self, df_1, title=None):
        x = PrettyTable()
        x.field_names = list(df_1.columns)
        df_1 = df_1.round(2)
        for i in range(len(df_1)):
            x.add_row(df_1.iloc[i])
        if title is not None:
            print(x.get_string(title=title))
        else:
            print(x)
            print("\n")
        return x
    
class Preprocessing(Pretty):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = None
    
    def load_dataset(self):
        df1 = pd.read_csv("data/aisles.csv")
        df2 = pd.read_csv("data/departments.csv")
        df3 = pd.read_csv("data/order_products__prior.csv")
        df4 = pd.read_csv("data/order_products__train.csv")
        df5 = pd.read_csv("data/orders.csv")
        df6 = pd.read_csv("data/products.csv")
        self.gen_table(df1.head(), "Aisles")
        print("\n")
        self.gen_table(df2.head(), "Departments")
        print("\n")
        self.gen_table(df3.head(), "Order Products Prior")
        print("\n")
        self.gen_table(df4.head(), "Order Products Train")
        print("\n")
        self.gen_table(df5.head(), "Orders")
        print("\n")
        self.gen_table(df6.head(), "Products")
        df5 = df5.fillna(0)
        df_train_prior = pd.concat([df3,df4],axis=0).sort_values(by="order_id")
        df_train_prior = pd.merge(left = df_train_prior, right = df6,
                             left_on='product_id', right_on='product_id').sort_values(by=['order_id']).reset_index(drop=True)
        df_train_prior = pd.merge(left = df_train_prior, right = df1,
                             left_on='aisle_id', right_on='aisle_id').sort_values(by=['order_id']).reset_index(drop=True)
        df_train_prior = pd.merge(left = df_train_prior, right = df2,
                                    left_on='department_id', right_on='department_id').sort_values(by=['order_id']).reset_index(drop=True)
        df_train_prior = pd.merge(left = df_train_prior, right = df5,
                                    left_on='order_id', right_on='order_id').sort_values(by=['order_id']).reset_index(drop=True)

        print(df_train_prior.dtypes, "Data Types")
        col_order = ['user_id','order_id','product_id','aisle_id','department_id','add_to_cart_order',
        'reordered','product_name','aisle','department','eval_set','order_number','order_dow','order_hour_of_day',
        'days_since_prior_order']

        df_train_prior = df_train_prior[col_order]
        #self.gen_table(df_train_prior.head(), "Train Prior")
        #random sampling
        self.dataset = df_train_prior.sample(200000, random_state=5805)
        self.gen_table(self.dataset.head(), "Downsampled Dataset")
    
    
    
        
c = Preprocessing()
print(c.load_dataset())