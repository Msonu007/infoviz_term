import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class Phase1():
    def __init__(self, dataset, dataset_full):
        self.dataset = pd.read_csv(dataset)
        print("downsample dataset loaded successfully")
        self.dataset_full = pd.read_csv(dataset_full)
        print("full sized dataset loaded sucessfully")
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
        fig.show()

    def plot_bar(self):
        # target variable distribution
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        sns.countplot(data=self.dataset, x="department", ax=ax)
        ax.set_xlabel("Department", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("Target variable distribution", fontdict=self.title_prop)
        ax.grid()
        fig.show()

    def plot_stacked_group_bar(self):
        temp_1 = self.dataset[self.dataset["reordered"] == 0]["product_name"].value_counts().reset_index().sort_values(by="product_name",ascending=False).iloc[:20].set_index(["index"]).to_dict()
        temp_2 = self.dataset[self.dataset["reordered"] == 1]["product_name"].value_counts().reset_index().sort_values(by="product_name",ascending=False).iloc[:20].set_index(["index"]).to_dict()
        temp_1 = temp_1["product_name"]
        temp_2 = temp_2["product_name"]
        temp_final = pd.DataFrame.from_dict({"product_name": temp_1.keys(), "ordered": temp_1.values(), "reordered": temp_2.values()})
        temp_final = temp_final.set_index(["product_name"])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        temp_final.plot(kind="bar", ax=ax)
        ax.set_xlabel("Product Name", fontdict=self.label_prop)
        ax.set_ylabel("Number of Orders", fontdict=self.label_prop)
        ax.set_title("Barplot Group for product name", fontdict=self.title_prop)
        fig.show()
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        temp_final.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Order Day of The Week", fontdict=self.label_prop)
        ax.set_ylabel("Number of Orders", fontdict=self.label_prop)
        ax.set_title("Barplot with stacks for Order_dow", fontdict=self.title_prop)
        fig.show()

    def count_plot(self):
        # number of aisles in the dataset
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.countplot(self.dataset, x="aisle_id", width=0.8, hue="department", dodge=False,
                      order=self.dataset["aisle_id"].value_counts().index, ax=ax)
        ax.set_title("histogram of various aisle's", fontdict=self.title_prop)
        ax.set_xlabel("aisle", fontdict=self.label_prop)
        ax.set_ylabel("frequency", fontdict=self.label_prop)
        ax.grid()
        fig.show()


    def plot_pie(self):
        # products sold in aisle reordered vs not reordered
        # fig,ax = plt.subplots(1,2)
        temp_1 = self.df[self.df["reordered"] == 1]["aisle"].value_counts().reset_index()
        temp_1.columns = ["aisle", "count"]
        temp_1 = temp_1.sort_values(by="count", ascending=False)
        temp_1 = temp_1.set_index("aisle")
        temp_k = temp_1.iloc[:20].__deepcopy__()
        temp_k["count"] = temp_k["count"] / temp_k["count"].sum()
        temp_k = list(temp_k["count"])
        # products sold in aisle reordered vs not reordered
        # fig,ax = plt.subplots(1,2)
        temp_2 = self.df[self.df["reordered"] == 0]["aisle"].value_counts().reset_index()
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
        fig.show()

    def plot_distplot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.distplot(self.df["order_hour_of_day"], ax=ax)
        ax.set_title("distplot of order_hour_of_day", fontdict=self.title_prop)
        ax.set_xlabel("Order_hour_of_day", fontdict=self.label_prop)
        ax.set_ylabel("Density", fontdict=self.label_prop)
        fig.show()

    def plot_pair_plot(self):
        ax = sns.pairplot(data=self.dataset[["order_dow", "days_since_prior_order", "order_hour_of_day"]])
        plt.show()

    def plot_heatmap_with_cbar(self):
        pass

    def plot_histogram_with_kde(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.histplot(data=new_df, x="user_id", kde=True, ax=ax, legend="user_id cart size")
        ax.set_xlabel("Cart Size", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("KDE of maximum cart size per order")
        ax.lines[0].set_color('r')
        fig.show()

    def plot_qq_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sm.qqplot(self.dataset["days_since_prior_order"], line='45', ax=ax)

        # Add labels and title
        ax.set_title('QQ Plot', fontdict=self.title_prop)
        ax.set_xlabel('Theoretical Quantiles', fontdict=self.label_prop)
        ax.set_ylabel('Sample Quantiles', fontdict=self.label_prop)

        # Show the plot
        fig.show()

    def plot_kde_alpha(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.kdeplot(data=new_df, x="user_id", ax=ax, fill=True, alpha=0.6)
        ax.set_xlabel("Cart Size", fontdict=self.label_prop)
        ax.set_ylabel("Frequency", fontdict=self.label_prop)
        ax.set_title("KDE of maximum cart size")
        fig.show()

    def plot_reg_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.scatterplot(data=self.dataset, x="order_dow", y="order_hour_of_day", ax=ax)
        sns.regplot(data=self.dataset, x="order_dow", y="order_hour_of_day", line_kws={"color": 'r'}, ax=ax)
        fig.show()

    def plot_box(self):
        pass

    def plot_area(self):
        pass

    def plot_violin(self):
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        sns.violinplot(data=self.dataset[["days_since_prior_order", "order_dow", "order_hour_of_day"]],ax=ax)
        ax.set_xlabel("numeric variables in the dataset")
        ax.set_title("Violin plot",fontdict=self.title_prop)
        fig.show()

    def plot_joint_kde(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day"]]
        ax = sns.jointplot(data=df_temp, x="order_dow", y="order_hour_of_day", kind="kde")
        ax.ax_joint.set_xlabel("order_dow", fontdict=self.label_prop)
        ax.ax_jointset_ylabel("order_hour_of_day", fontdict=self.label_prop)
        ax.ax_jointset_title("plot_joint_kde", fontdict=self.title_prop)

    def plot_joint_scatter(self):
        df_temp = self.dataset[["order_dow", "order_hour_of_day"]]
        ax = sns.jointplot(data=df_temp, x="order_dow", y="order_hour_of_day", kind="scatter")
        ax.ax_joint.set_xlabel("order_dow", fontdict=self.label_prop)
        ax.ax_jointset_ylabel("order_hour_of_day", fontdict=self.label_prop)
        ax.ax_jointset_title("plot_joint_kde", fontdict=self.title_prop)

    def plot_rug_plot(self):
        new_df = self.dataset_full["user_id"].value_counts().reset_index()
        sns.rugplot(x='user_id', data=new_df)
        sns.histplot(data=new_df, x="user_id", kde=True, fill=False)
        plt.show()

    def threeD_contour(self):
        pass

    def plot_cluster_map(self):
        pass

    def plot_hexbin(self):
        df_temp = self.dataset[["order_dow", "days_since_prior_order"]]
        grid_ = sns.jointplot(data=df_temp, x="order_dow", y="days_since_prior_order", kind="hex")
        grid_.ax_joint.set_title("Hexbin Plot",fontdict=self.title_prop)

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

    def plot_swarm(self):
        pass


c = Phase1("data/downsample.csv", "data/train_final.csv")
c.plot_bar()
