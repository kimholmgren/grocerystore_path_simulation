import pandas as pd
import numpy as np
import os

class ShoppingListGen:
    """
    This shopping list creation is based on the 2017 Instacart Grocery Shopping
    Dataset. It was temporarily unavailable when I attempted to access it,
    so I used the version from the Kaggle link below. The data is not included in
    this repository because of the size and because I don't own it, but it should be
    accessible at the links below. The code is based on the structure of the
    dataset as I have it from Kaggle. This class could be replaced with another
    one to use another dataset, as long as it has a generator function.


    Psparks. “Instacart Market Basket Analysis.” Kaggle, 20 Nov. 2017,
    www.kaggle.com/psparks/instacart-market-basket-analysis.

    “The Instacart Online Grocery Shopping Dataset 2017.” Instacart,
    www.instacart.com/datasets/grocery-shopping-2017.

    """
    MY_DEPTS = ['produce', 'grocery', 'meatdairy', 'household', 'frozen', 'checkout', None]
    MY_DPT_MAP = [4, 6, 1, 0, 6, 1, 1, 3, 1, 1, 3, 2, 1, 1, 1, 2, 3, 3, 1, 2, 6]


    def __init__(self, dir, dpt_names = MY_DEPTS, dpt_map = MY_DPT_MAP,
                 seed=None):
        """
        Initialize the shopping list generator with a directory to find data
        :param dir: directory to find data
        :param dpt_names: name of departments in this layout, in order
        :param dpt_map: mapping from departments.csv to departments in legend by
        index, where maxindex+1 signifies "None", or don't use products from
        this department
        """
        self.dir = dir
        self.read_dfs()
        # map departments
        self.dpt_names = dpt_names
        self.dpts["mapped_dpt"] = dpt_map
        self.dpts["mapped_dpt_name"] = np.array(dpt_names)[dpt_map]
        # map products
        self.prods = self.prods.merge(self.dpts, on="department_id")
        self.prods = self.prods[
            ['product_id', 'product_name', 'mapped_dpt', 'mapped_dpt_name']]
        # map orders
        self.orders = self.orders.merge(self.prods, on="product_id")[
        ['order_id', 'product_id', 'product_name', 'mapped_dpt',
         'mapped_dpt_name']]
        # get unique orders
        self.order_ids = self.orders['order_id'].unique()
        # set seed
        if seed is not None:
            np.random.seed(seed)

    def read_dfs(self):
        """
        Read the department, product, and order csvs from the directory
        :return: None, dataframes are stored in self
        """
        self.dpts = pd.read_csv(os.path.join(self.dir, "departments.csv"))
        self.prods = pd.read_csv(os.path.join(self.dir, "products.csv"))
        self.orders = pd.read_csv(
            os.path.join(self.dir, "order_products__train.csv"))
        return

    def __iter__(self):
        return self

    def __next__(self):
        """
        Randomly select one of the shopping lists
        :return: df containing a unique order, all products, and the remapped
        departments to find those products in
        """
        curr_id = np.random.choice(self.order_ids)
        curr_order = self.orders[self.orders['order_id']==curr_id]
        curr_order = curr_order[curr_order['mapped_dpt']!=len(self.dpt_names)-1]
        return curr_order