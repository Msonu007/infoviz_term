import prettytable
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import copy
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


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
        self.gen_table(df_train_prior.head(), "Train Prior")
        final_df = df_train_prior[df_train_prior["reordered"] == 0].sample(100000, random_state=5805)
        final_df = pd.concat(
            [final_df, df_train_prior[df_train_prior["reordered"] == 1].sample(100000, random_state=5805)])
        final_df.to_csv("downsample.csv", index=False)
        df_train_prior.to_csv("train_final.csv")
        self.dataset = df_train_prior
        self.gen_table(self.dataset.head(), "Downsampled Dataset")
    def statstics(self):
        print(self.dataset[["order_dow","days_since_prior_order","order_hour_of_day","add_to_cart_order","order_number"]].describe())    
    
    def perform_pca(self):
        pca = PCA()
        self.dataset = pd.read_csv("downsample,csv")
        self.X_train = self.dataset[["order_dow","days_since_prior_order","order_hour_of_day","add_to_cart_order","order_number"]]
        X_train_pca = pca.fit_transform(self.X_train)
        print(f"The condition number of the dataset before pca {np.linalg.cond(self.X_train)}")
        pca_ = PCA(1)
        print(f"The condition number of the dataset after pca {np.linalg.cond(pca_.fit_transform(self.X_train))}")
        exp_V = pca.explained_variance_ratio_
        print(exp_V)
        sum_val = 0
        index_ = None
        sum_ = []
        for i in range(len(exp_V)):
            sum_val += exp_V[i]
            sum_.append(sum_val)
            if sum_val >= 0.9:
                if index_ is None:
                    index_ = i + 1
        print("n_features are ", index_)
        plt.plot(list(range(1, len(exp_V) + 1)), np.cumsum(exp_V), marker=".")
        plt.yticks(np.linspace(0, 1, 11))
        plt.plot([index_] * 10, np.linspace(0.1, 1, 10))
        plt.plot(range(1, 26), [0.9] * 25)
        plt.xticks(range(1, len(exp_V) + 1))
        plt.xlabel("n_features", fontdict=self.label_prop)
        plt.ylabel("cumulative explained variance", fontdict=self.label_prop)
        plt.title("PCA", fontdict=self.title_prop)
        plt.show()

    def perform_rfc(self):
        c = RandomForestClassifier(random_state=5805)
        self.X_train = self.dataset[["order_dow","days_since_prior_order","order_hour_of_day","add_to_cart_order","order_number"]]
        self.y_train = self.dataset["reordered"]
        c.fit(self.X_train,self.y_train)
        f_imp = list(c.feature_importances_)
        final_f_imp = sorted([[list(self.X_train.columns)[i], f_imp[i]] for i in range(len(f_imp))], reverse=True,
                             key=lambda x: x[1])
        sum_val = 0
        index_ = None
        sum_ = []
        for i in range(len(final_f_imp)):
            sum_val += final_f_imp[i][1]
            sum_.append(sum_val)
            if sum_val >= 0.95:
                if index_ is None:
                    index_ = i
        plt.plot(range(1, len(sum_) + 1), sum_, linewidth=1.5, marker='.')
        plt.plot([index_ + 1] * 50, np.linspace(0.1, 1, 50), linewidth=0.5)
        plt.plot(list(range(1, 27)), [0.95] * 26, linewidth=0.5)
        plt.xticks(range(1, len(sum_) + 1))
        plt.yticks(np.linspace(0, 1, 21))
        plt.xlabel("Number of Features", fontdict=self.label_prop)
        plt.ylabel("Feature importances", fontdict=self.label_prop)
        plt.title("rfc_feature_extraction", fontdict=self.title_prop)
        plt.show()
        feature_importance = pd.DataFrame({
            'Feature': list(self.X_train.columns),
            'feature_importance': f_imp
        }).sort_values(by='feature_importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='feature_importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance from Random Forest', fontdict=self.title_prop)
        plt.xlabel('Importance', fontdict=self.label_prop)
        plt.ylabel('Feature', fontdict=self.label_prop)
        plt.show()
        self.rfr_imp = [final_f_imp[i][0] for i in range(index_ + 1)]
        return self.rfr_imp
    def box_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.boxplot(data=self.X_train, ax=ax)
        plt.show()

        
c = Preprocessing()
print(c.load_dataset())