import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math
import time
import copy
import scipy.stats as stats
import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from plotly.subplots import make_subplots

title_font_size = 20
title_font_family = "serif"
title_font_color = "blue"
label_font_size = 15
label_font_color = "darkred"
label_font_family = "serif"

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
        self.label_prop = {"font": "serif", "color": "darkred", "size": 15}
        self.title_prop = {"font": "serif", "color": "blue", "size": 20}

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
        nan_rows = df5[df5.isnull().any(axis=1)]
        calls = df5.loc[nan_rows.index]
        self.gen_table(calls.head(), "Dataset before cleaning")
        df5 = df5.fillna(0)
        self.gen_table(df5.loc[nan_rows.index].head(), "Dataset after cleaning")
        if not os.path.exists("train_final.csv"):
            df_train_prior = pd.concat([df3, df4], axis=0).sort_values(by="order_id")
            df_train_prior = pd.merge(left=df_train_prior, right=df6,
                                      left_on='product_id', right_on='product_id').sort_values(by=['order_id']).reset_index(
                drop=True)
            df_train_prior = pd.merge(left=df_train_prior, right=df1,
                                      left_on='aisle_id', right_on='aisle_id').sort_values(by=['order_id']).reset_index(
                drop=True)
            df_train_prior = pd.merge(left=df_train_prior, right=df2,
                                      left_on='department_id', right_on='department_id').sort_values(
                by=['order_id']).reset_index(drop=True)
            df_train_prior = pd.merge(left=df_train_prior, right=df5,
                                      left_on='order_id', right_on='order_id').sort_values(by=['order_id']).reset_index(
                drop=True)

            print(df_train_prior.dtypes, "Data Types")
            col_order = ['user_id', 'order_id', 'product_id', 'aisle_id', 'department_id', 'add_to_cart_order',
                         'reordered', 'product_name', 'aisle', 'department', 'eval_set', 'order_number', 'order_dow',
                         'order_hour_of_day',
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
        print(self.dataset[["order_dow", "days_since_prior_order", "order_hour_of_day", "add_to_cart_order",
                            "order_number"]].describe())

    def perform_pca(self):
        pca = PCA()
        self.dataset = pd.read_csv("downsample.csv")
        self.X_train = self.dataset[
            ["order_dow", "days_since_prior_order", "order_hour_of_day", "add_to_cart_order", "order_number"]]
        X_train_pca = pca.fit_transform(self.X_train)
        print(f"The condition number of the dataset before pca {np.linalg.cond(self.X_train)}")
        pca_ = PCA(3)
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
        self.X_train = self.dataset[
            ["order_dow", "days_since_prior_order", "order_hour_of_day", "add_to_cart_order", "order_number"]]
        self.y_train = self.dataset["reordered"]
        c.fit(self.X_train, self.y_train)
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
class Phase1():
    def __init__(self, dataset, dataset_full):
        self.dataset = pd.read_csv(dataset)
        self.c1 = Pretty()
        self.c1.gen_table(self.dataset.head())
        print("downsample dataset loaded successfully")
        #self.dataset_full = pd.read_csv(dataset_full)
        #print("full sized dataset loaded sucessfully")
        self.label_prop = {"font": "serif", "color": "darkred", "size": 15}
        self.title_prop = {"font": "serif", "color": "blue", "size": 20}

    def plot_line(self):
        order_dow_count = self.dataset["order_dow"].value_counts().reset_index().sort_values(by="index")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.plot(order_dow_count["index"], order_dow_count["order_dow"])
        # ax.set_xticks(["Sun","Mon","Tue","Wed","Thu","Fri","Sat"])
        ax.set_xlabel("Day of the week", fontdict=self.label_prop)
        ax.set_ylabel("Number of orders placed", fontdict=self.label_prop)
        ax.set_title("Line plot n_orders vs dow", fontdict=self.title_prop)
        plt.show()

    def plot_bar(self):
        # target variable distribution
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        top_10_categories = self.dataset['department'].value_counts().index[:20]

        df_top_10 = self.dataset[self.dataset['department'].isin(top_10_categories)]

        sns.countplot(x='department', data=df_top_10, order=df_top_10['department'].value_counts().index)
        ax.set_xlabel("Department", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("Top 20 departments bar plot", fontdict=self.title_prop)
        ax.grid()
        plt.show()

    def plot_stacked_group_bar(self):
        def make_data(col):
            temp_1 = self.dataset[self.dataset["reordered"] == 0][col].value_counts().reset_index().sort_values(
                by=col, ascending=False).iloc[:20].set_index(["index"]).to_dict()
            temp_2 = self.dataset[self.dataset["reordered"] == 1][col].value_counts().reset_index().sort_values(
                by=col, ascending=False).iloc[:20].set_index(["index"]).to_dict()
            temp_1 = temp_1[col]
            temp_2 = temp_2[col]
            temp_final = pd.DataFrame.from_dict(
                {col: temp_1.keys(), "ordered": temp_1.values(), "reordered": temp_2.values()})
            temp_final = temp_final.set_index([col])
            return temp_final
        temp_final = make_data("product_name")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        temp_final.plot(kind="bar", ax=ax)
        ax.set_xlabel("Product Name", fontdict=self.label_prop)
        ax.set_ylabel("Number of Orders", fontdict=self.label_prop)
        ax.set_title("Barplot Group for product name", fontdict=self.title_prop)
        plt.show()
        temp_final = make_data("order_dow")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        temp_final.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Order Day of The Week", fontdict=self.label_prop)
        ax.set_ylabel("Number of Orders", fontdict=self.label_prop)
        ax.set_title("Barplot with stacks for Order_dow", fontdict=self.title_prop)
        plt.show()

    def count_plot(self):
        # number of aisles in the dataset
        top_10_categories = self.dataset['aisle_id'].value_counts().index[:20]

        df_top_10 = self.dataset[self.dataset['aisle_id'].isin(top_10_categories)]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.countplot(df_top_10, x="aisle_id", width=0.8, hue="department", dodge=False,
                      order=df_top_10["aisle_id"].value_counts().index, ax=ax)
        ax.set_title("Countplot of various aisle's", fontdict=self.title_prop)
        ax.set_xlabel("aisle", fontdict=self.label_prop)
        ax.set_ylabel("frequency", fontdict=self.label_prop)
        ax.grid()
        plt.show()

    def plot_pie(self):
        # products sold in aisle reordered vs not reordered
        # fig,ax = plt.subplots(1,2)
        temp_1 = self.dataset[self.dataset["reordered"] == 1]["aisle"].value_counts().reset_index()
        temp_1.columns = ["aisle", "count"]
        temp_1 = temp_1.sort_values(by="count", ascending=False)
        temp_1 = temp_1.set_index("aisle")
        temp_k = temp_1.iloc[:20].__deepcopy__()
        temp_k["count"] = temp_k["count"] / temp_k["count"].sum()
        temp_k = list(temp_k["count"])
        # products sold in aisle reordered vs not reordered
        # fig,ax = plt.subplots(1,2)
        temp_2 = self.dataset[self.dataset["reordered"] == 0]["aisle"].value_counts().reset_index()
        temp_2.columns = ["aisle", "count"]
        temp_2 = temp_2.sort_values(by="count", ascending=False)
        temp_2 = temp_2.set_index("aisle")
        temp_k2 = temp_2.iloc[:20].__deepcopy__()
        temp_k2["count"] = temp_k2["count"] / temp_k2["count"].sum()
        temp_k2 = list(temp_k2["count"])
        explode_ = []
        for i in range(len(temp_k)):
            if temp_k[i] < 0.02:
                explode_.append(0.1)
            else:
                explode_.append(0)
        font_pie = {"size": 10}
        fig, ax = plt.subplots(1, 2, figsize=(36, 18))
        plt.figure()
        temp_1.sort_values(by="count", ascending=False).iloc[:20].plot(
            kind="pie",
            y="count",
            autopct=lambda x: np.round(x, 2),
            explode=tuple(explode_),
            textprops=font_pie,
            ax=ax[0],
        )
        ax[0].legend(loc="upper right")
        ax[0].set_title("reordered items", fontdict=self.title_prop)
        explode_ = []
        for i in range(len(temp_k2)):
            if temp_k2[i] < 0.02:
                explode_.append(0.1)
            else:
                explode_.append(0)
        font_pie = {"size": 10}
        temp_2.sort_values(by="count", ascending=False).iloc[:20].plot(
            kind="pie",
            y="count",
            autopct=lambda x: np.round(x, 2),
            explode=tuple(explode_),
            textprops=font_pie,
            ax=ax[1],
        )
        ax[1].legend(loc="upper right")
        ax[1].set_title("not reordered items", fontdict=self.title_prop)
        plt.show()

    def plot_distplot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.distplot(np.sqrt(self.dataset["order_hour_of_day"]), ax=ax)
        ax.set_title("distplot of order_hour_of_day", fontdict=self.title_prop)
        ax.set_xlabel("Order_hour_of_day", fontdict=self.label_prop)
        ax.set_ylabel("Density", fontdict=self.label_prop)
        plt.show()

    def plot_pair_plot(self):
        ax = sns.pairplot(
            data=self.dataset[["order_dow", "days_since_prior_order", "order_hour_of_day"]].apply(np.sqrt))
        plt.show()

    def plot_heatmap_with_cbar(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day", "days_since_prior_order",
                                "add_to_cart_order"]].applymap(np.sqrt)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.heatmap(np.round(df_temp.corr(), 2), annot=True, cmap="viridis", ax=ax)
        ax.set_title("Heatmap with colorbar", fontdict=self.title_prop)
        plt.show()

    def plot_histogram_with_kde(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.histplot(data=new_df, x="user_id", kde=True, ax=ax, legend="user_id cart size")
        ax.set_xlabel("Cart Size", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("KDE of maximum cart size per order")
        ax.lines[0].set_color('r')
        plt.show()

    def plot_qq_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sm.qqplot(self.dataset["days_since_prior_order"].map(np.sqrt), line='45', ax=ax)

        # Add labels and title
        ax.set_title('QQ Plot', fontdict=self.title_prop)
        ax.set_xlabel('Theoretical Quantiles', fontdict=self.label_prop)
        ax.set_ylabel('Sample Quantiles', fontdict=self.label_prop)

        # Show the plot
        plt.show()

    def plot_kde_alpha(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.kdeplot(data=new_df, x="user_id", ax=ax, fill=True, alpha=0.6)
        ax.set_xlabel("Cart Size", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("KDE of maximum cart size",fontdict=self.title_prop)
        plt.show()

    def plot_reg_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        df = self.dataset[["order_dow", "order_hour_of_day"]].applymap(np.sqrt)
        sns.scatterplot(data=df, x="order_dow", y="order_hour_of_day", ax=ax)
        sns.regplot(data=df, x="order_dow", y="order_hour_of_day", line_kws={"color": 'r'}, ax=ax)
        ax.set_xlabel("order_dowe", fontdict=self.label_prop)
        ax.set_ylabel("Order Hour of the Day", fontdict=self.label_prop)
        ax.set_title("Reg plot for Order_dow and Order_hour_of_day")
        plt.show()

    def plot_box(self):
        pass

    def plot_area(self):
        df = self.dataset["order_dow"].value_counts().reset_index()
        df.columns = ["order_dow", "count"]
        df = df.sort_values(by="order_dow")
        print(df.head())

        # Create a line plot
        sns.lineplot(data=df, x='order_dow', y='count', label="count")

        # Get the current Axes instance
        ax = plt.gca()
        for i in range(len(df)):
            ax.plot([df["order_dow"]] * 1000, np.linspace(0, df["count"], 1000), color="red", alpha=0.4, linestyle="--")
        # Fill the area under the line
        ax.fill_between(df['order_dow'], df['count'], color='skyblue', alpha=0.4, label="area")
        ax.set_xlabel("order_dow", fontdict=self.label_prop)
        ax.set_ylabel("count", fontdict=self.label_prop)
        ax.set_title("Area plot", fontdict=self.title_prop)
        ax.legend()

        # Show the plot
        plt.show()

    def plot_violin(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.violinplot(data=self.dataset[["days_since_prior_order", "order_dow", "order_hour_of_day"]], ax=ax)
        ax.set_xlabel("numeric variables in the dataset")
        ax.set_title("Violin plot", fontdict=self.title_prop)
        plt.show()

    def plot_joint_kde(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day"]].applymap(np.sqrt)
        ax = sns.jointplot(data=df_temp, x="order_dow", y="order_hour_of_day", kind="kde")
        ax.ax_joint.set_xlabel("order_dow", fontdict=self.label_prop)
        ax.ax_joint.set_ylabel("order_hour_of_day", fontdict=self.label_prop)
        ax.ax_joint.set_title("plot_joint_kde", fontdict=self.title_prop)
        plt.show()

    def plot_joint_scatter(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day"]].applymap(np.sqrt)
        ax = sns.jointplot(data=df_temp, x="order_dow", y="order_hour_of_day", kind="scatter")
        ax.ax_joint.set_xlabel("order_dow", fontdict=self.label_prop)
        ax.ax_joint.set_ylabel("order_hour_of_day", fontdict=self.label_prop)
        ax.ax_joint.set_title("plot_joint_kde", fontdict=self.title_prop)
        plt.show()

    def plot_rug_plot(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        sns.rugplot(x='user_id', data=new_df)
        sns.histplot(data=new_df, x="user_id", kde=True, fill=False)
        plt.xlabel("user_id",fontdict=self.label_prop)
        plt.ylabel("Count",fontdict=self.label_prop)
        plt.title("Rug plot",fontdict=self.title_prop)
        plt.show()

    def threeD(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day", "days_since_prior_order"]].applymap(np.sqrt)
        # Create grid and multivariate normal
        x = np.linspace(min(df_temp["order_dow"]), max(df_temp["order_dow"]), 500)
        y = np.linspace(min(df_temp["order_hour_of_day"]), max(df_temp["order_hour_of_day"]), 500)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X;
        pos[:, :, 1] = Y
        rv = multivariate_normal([df_temp["order_dow"].mean(), df_temp["order_hour_of_day"].mean()], [[1, 0], [0, 1]])

        # Make a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
        ax.set_xlabel('order_dow')
        ax.set_ylabel('order_hour_of_day')
        ax.set_zlabel('PDF')
        ax.set_title('3D contour plot')
        plt.show()

    def plot_contour(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day", "days_since_prior_order"]].applymap(np.sqrt)
        plt.figure(figsize=(10, 10))
        sns.kdeplot(data=self.dataset, x="order_dow", y="order_hour_of_day", fill=True, alpha=0.6, cmap="viridis")
        plt.xlabel("order_dow", fontdict=self.label_prop)
        plt.ylabel("order_hour_of_day", fontdict=self.label_prop)
        plt.title("Contour plot", fontdict=self.title_prop)
        plt.show()

    def plot_cluster_map(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day", "days_since_prior_order"]].applymap(np.sqrt)

        # Create a cluster map
        cluster_map = sns.clustermap(np.round(df_temp.corr(),2), annot=True, cmap='viridis')
        plt.suptitle('ClusterGrid ', fontdict=self.title_prop)
        # Show the plot
        plt.show()

    def plot_hexbin(self):
        df_temp = self.dataset[["order_dow", "days_since_prior_order"]]
        grid_ = sns.jointplot(data=df_temp, x="order_dow", y="days_since_prior_order", kind="hex")
        grid_.ax_joint.set_title("Hexbin Plot", fontdict=self.title_prop)
        plt.show()

    def plot_strip(self):
        synthetic_data = self.dataset["order_id"].value_counts().reset_index()
        order_dow = []
        for i in range(len(synthetic_data)):
            order_id = synthetic_data["index"].iloc[i]
            order_dow.append(self.dataset[self.dataset["order_id"] == order_id]["order_dow"].unique()[0])
        temp_ = self.dataset["order_id"].value_counts().reset_index()
        temp_["order_dow"] = order_dow
        ax = sns.stripplot(data=temp_, x="order_dow", y="order_id")
        ax.set_yticks([1, 2, 3, 4])
        ax.set_xlabel("Order_dow",fontdict=self.label_prop)
        ax.set_ylabel("User id",fontdict=self.label_prop)
        ax.set_title("Strip plot")
        plt.show()

    def plot_subplots(self):
        fig,ax = plt.subplots(2,2,figsize=(10,10))
        top_10_categories = self.dataset['aisle_id'].value_counts().index[:20]

        df_top_10 = self.dataset[self.dataset['aisle_id'].isin(top_10_categories)]
        sns.countplot(df_top_10, x="aisle_id", width=0.8, hue="department", dodge=False,
                      order=df_top_10["aisle_id"].value_counts().index, ax=ax[0][0])
        ax[0][0].set_title("Countplot of top 20 aisle's", fontdict=self.title_prop)
        ax[0][0].set_xlabel("aisle", fontdict=self.label_prop)
        ax[0][0].set_ylabel("frequency", fontdict=self.label_prop)
        last_10_categories = self.dataset['aisle_id'].value_counts().index[-20:-1]

        df_last_10 = self.dataset[self.dataset['aisle_id'].isin(last_10_categories)]
        sns.countplot(df_last_10, x="aisle_id", width=0.8, hue="department", dodge=False,
                      order=df_last_10["aisle_id"].value_counts().index, ax=ax[0][1])
        ax[0][1].set_title("Countplot of least 20 aisle's", fontdict=self.title_prop)
        ax[0][1].set_xlabel("aisle", fontdict=self.label_prop)
        ax[0][1].set_ylabel("frequency", fontdict=self.label_prop)
        ax[0][1].grid()
        top_10_categories = self.dataset['user_id'].value_counts().index[:20]

        df_top_10 = self.dataset[self.dataset['user_id'].isin(top_10_categories)]
        sns.countplot(df_top_10, x="user_id", width=0.8, hue="department", dodge=False,
                      order=df_top_10["user_id"].value_counts().index, ax=ax[1][0])
        ax[1][0].set_title("Countplot of top 20 users's", fontdict=self.title_prop)
        ax[1][0].set_xlabel("users", fontdict=self.label_prop)
        ax[1][0].set_ylabel("frequency", fontdict=self.label_prop)
        last_10_categories = self.dataset['user_id'].value_counts().index[-20:-1]

        df_last_10 = self.dataset[self.dataset['user_id'].isin(last_10_categories)]
        sns.countplot(df_last_10, x="user_id", width=0.8, hue="department", dodge=False,
                      order=df_last_10["user_id"].value_counts().index, ax=ax[1][1])
        ax[1][1].set_title("Countplot of least 20 user's", fontdict=self.title_prop)
        ax[1][1].set_xlabel("users", fontdict=self.label_prop)
        ax[1][1].set_ylabel("frequency", fontdict=self.label_prop)
        ax[1][1].grid()
        plt.show()

        fig,ax = plt.subplots(2,2,figsize=(10,10))
        sns.distplot(np.sqrt(self.dataset[self.dataset["reordered"] == 0]["order_hour_of_day"]), ax=ax[0][0])
        ax[0][0].set_title("distplot of order_hour_of_day that are not reordered", fontdict=self.title_prop)
        ax[0][0].set_xlabel("Order_hour_of_day", fontdict=self.label_prop)
        ax[0][0].set_ylabel("Density", fontdict=self.label_prop)
        sns.distplot(np.sqrt(self.dataset[self.dataset["reordered"] == 1]["order_hour_of_day"]), ax=ax[0][1])
        ax[0][1].set_title("distplot of order_hour_of_day that are reordered", fontdict=self.title_prop)
        ax[0][1].set_xlabel("Order_hour_of_day", fontdict=self.label_prop)
        ax[0][1].set_ylabel("Density", fontdict=self.label_prop)
        sns.distplot(np.sqrt(self.dataset[self.dataset["reordered"] == 0]["order_dow"]), ax=ax[1][0])
        ax[1][0].set_title("distplot of order_dow that are not reordered", fontdict=self.title_prop)
        ax[1][0].set_xlabel("Order_day_of_week", fontdict=self.label_prop)
        ax[1][0].set_ylabel("Density", fontdict=self.label_prop)
        sns.distplot(np.sqrt(self.dataset[self.dataset["reordered"] == 1]["order_dow"]), ax=ax[1][1])
        ax[0][1].set_title("distplot of order_dow that are reordered", fontdict=self.title_prop)
        ax[0][1].set_xlabel("Order_dow", fontdict=self.label_prop)
        ax[0][1].set_ylabel("Density", fontdict=self.label_prop)
        plt.show()
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        df = self.dataset["order_dow"].value_counts().reset_index()
        df.columns = ["order_dow", "count"]
        df = df.sort_values(by="order_dow")
        # Create a line plot
        sns.lineplot(data=df, x='order_dow', y='count', label="count",ax=ax[0])
        # Get the current Axes instance
        for i in range(len(df)):
            ax[0].plot([df["order_dow"]] * 1000, np.linspace(0, df["count"], 1000), color="red", alpha=0.4, linestyle="--")
        # Fill the area under the line
        ax[0].fill_between(df["order_dow"], df['count'], color='skyblue', alpha=0.4, label="area")
        ax[0].set_xlabel("order_dow", fontdict=self.label_prop)
        ax[0].set_ylabel("count", fontdict=self.label_prop)
        ax[0].set_title("Area plot", fontdict=self.title_prop)
        ax[0].legend()
        df = self.dataset["order_hour_of_day"].value_counts().reset_index()
        df.columns = ["order_hour_of_day", "count"]
        df = df.sort_values(by="order_hour_of_day")
        # Create a line plot
        sns.lineplot(data=df, x='order_hour_of_day', y='count', label="count", ax=ax[1])
        # Get the current Axes instance
        for i in range(len(df)):
            ax[1].plot([df["order_hour_of_day"]] * 1000, np.linspace(0, df["count"], 1000), color="red", alpha=0.4,
                       linestyle="--")
        # Fill the area under the line
        ax[1].fill_between(df["order_hour_of_day"], df['count'], color='skyblue', alpha=0.4, label="area")
        ax[1].set_xlabel("order_hour_of_day", fontdict=self.label_prop)
        ax[1].set_ylabel("count", fontdict=self.label_prop)
        ax[1].set_title("Area plot", fontdict=self.title_prop)
        ax[1].legend()
        plt.show()

    def plot_swarm(self):
        pass

    def plot_all(self):
        self.plot_line()
        self.count_plot()
        self.plot_area()
        self.plot_bar()
        self.plot_box()
        self.plot_cluster_map()
        self.plot_contour()
        self.plot_distplot()
        self.plot_heatmap_with_cbar()
        self.plot_hexbin()
        self.plot_histogram_with_kde()
        self.plot_joint_kde()
        self.plot_joint_scatter()
        self.plot_kde_alpha()
        self.plot_pair_plot()
        self.plot_pie()
        self.plot_qq_plot()
        self.plot_reg_plot()
        self.plot_rug_plot()
        self.plot_stacked_group_bar()
        self.plot_strip()
        self.plot_violin()
        self.plot_subplots()
        self.plot_contour()
        self.threeD()

c1 = Preprocessing()
c1.load_dataset()
c1.perform_pca()
c1.perform_rfc()
c1.box_plot()
c = Phase1("downsample.csv", "train_final.csv")
c.plot_subplots()
