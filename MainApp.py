import tkinter as tk
from tkinter import font as tkFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import tree
from collections import defaultdict
import random
import pydot
from io import StringIO
import pydotplus
from multiprocessing import Process
import tkinter as tk
from tkinter import filedialog
import kmedoid as kmd
from tkinter import messagebox
from tkinter import StringVar

import pandas

class MovieApp:
    def __init__(self, master, dataframe,clusters,edit_rows=[]):
        self.master = master
        self.frame = tk.Frame(self.master)
        #self.button1 = tk.Button(self.frame, text = 'New Window', width = 25,command=self.plot_graph)
        #self.button1.pack()
        #self.button2 = tk.Button(self.frame, text = 'PLEASE PROCESS', width = 25, command = self.process_data_set)
        #self.button2.pack()
        self.button3 = tk.Button(self.frame, text = 'DETAILED REPORT', width = 25, command = self.show_results)
        self.button3.pack()
        self.button4 = tk.Button(self.frame, text = 'Add a Movie', width = 25, command = self.makeform)
        self.button4.pack()

    
        self.frame.pack()
        #       the dataframe
        self.df = dataframe
        self.clusters = clusters
        self.dat_cols = list(self.df)
        if edit_rows:
            self.dat_rows = edit_rows
        else:
            self.dat_rows = range(len(self.df))
        self.rowmap = {i: row for i, row in enumerate(self.dat_rows)}

#       subset the data and convert to giant list of strings (rows) for viewing
        self.sub_data = self.df.ix[self.dat_rows, self.dat_cols]
        self.sub_datstring = self.sub_data.to_string().split('\n')
        self.title_string = self.sub_datstring[0]
        self.results = ""
        self.clicked=0

# save the format of the lines, so we can update them without re-running
# df.to_string()
        self._get_line_format(self.title_string)

        self.update_history = []
        self.movie_t = StringVar()
        self.no_usr_rev = StringVar()
        self.budget = StringVar()
        self.no_critic_reviews = StringVar()
        self.fb_likes = StringVar()
        self.usr_votes = StringVar()
        self.duration = StringVar()

        self.tree = None

    def _rewrite(self):
        """ re-writing the dataframe string in the listbox"""
        new_col_vals = self.df.ix[self.row, self.dat_cols].astype(str).tolist()
        new_line = self._make_line(new_col_vals)
        if self.lb.cget('state') == tk.DISABLED:
            self.lb.config(state=tk.NORMAL)
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)
            self.lb.config(state=tk.DISABLED)
        else:
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)

    def _get_line_format(self, line):
        """ save the format of the title string, stores positions
            of the column breaks"""
        pos = [1 + line.find(' ' + n) + len(n) for n in self.dat_cols]
        self.entry_length = [pos[0]] + \
            [p2 - p1 for p1, p2 in zip(pos[:-1], pos[1:])]

    def _make_line(self, col_entries):
        """ add a new line to the database in the correct format"""
        new_line_entries = [('{0: >%d}' % self.entry_length[i]).format(entry)
                            for i, entry in enumerate(col_entries)]
        new_line = "".join(new_line_entries)
        return new_line
    
    '''def plot_graph(self):
        markers = ['bo', 'go', 'ro', 'c+', 'm+', 'y+']
        clusters = self.clusters
        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                plt.plot(df[0], df[1], markers[i])
        plt.xlabel('IMDb Scores')
        plt.ylabel('Gross')
        plt.title('K-medoid clusters')
        plt.legend()
        plt.show()'''
    
    def assign_target(self,row):

        x = row['movie_title']
        clusters = self.clusters
        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                if df[2] == x:
                    row['cluster'] = 'cluster'+str(i)

        return row

    def show_results(self):
        #choosing features for decision tree
        columns = [ 'movie_title','num_user_for_reviews', 'budget'
                    , 'num_critic_for_reviews','movie_facebook_likes','num_voted_users','duration']
        
        df = self.df[columns]
        df = df.apply(self.assign_target, axis=1)
        df.drop(labels = ['movie_title'], axis = 1, inplace = True)

        #creating training and test sets
        splitSet = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=0)

        for train_index, test_index in splitSet.split(df, df['cluster']):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        Y_train = train_set.cluster
        X_train = train_set[train_set.columns.drop('cluster')]
        Y_test = test_set.cluster
        X_test = test_set[test_set.columns.drop('cluster')]

        #Creating decision tree 
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        self.tree = decision_tree
        predictions = decision_tree.predict(X_test)

        output = 'Score of the decision tree='+str(decision_tree.score(X_test, Y_test))+('\n')

        output = output+'\nDecision Tree Confusion Matrix\n\n'+str(confusion_matrix(Y_test,predictions))+('\n')
       
        output = output+'\nDecision Tree Classification Report\n\n'+str(classification_report(Y_test,predictions))+('\n')

        #Applying random forest classifier
        rfc = RandomForestClassifier(n_estimators=2000)
        rfc.fit(X_train, Y_train)
        output = output+('Random Forest Statistics\n')

        rfc_pred = rfc.predict(X_test)
        output = output+'\nRandom Forest Confusion Matrix\n\n'+str(confusion_matrix(Y_test,rfc_pred))+('\n')
        
        output = output+'\nRandom Forest Classification Report\n\n'+str(classification_report(Y_test,rfc_pred))+('\n')
       
        print(output)
        self.clicked +=1
        self.results = output


        messagebox.showinfo("Results of the decision tree model",self.results)
        
        
        
    def makeform(self):

        if self.clicked ==0:
            messagebox.showerror("Please process the clusters","Please press the process button before this")
        else:
            master = tk.Toplevel(self.master)
            master.geometry("350x350")

            tk.Label(master, text="Movie Title").grid(row=0,columnspan=10)
            tk.Label(master, text="Number of user reviews").grid(row=1,columnspan=10)
            tk.Label(master, text="Budget").grid(row=2,columnspan=10)
            tk.Label(master, text="Number of critic reviews").grid(row=3,columnspan=10)
            tk.Label(master, text="Movie facebook likes").grid(row=4,columnspan=10)
            tk.Label(master, text="Number of user votes").grid(row=5,columnspan=10)
            tk.Label(master, text="Duration").grid(row=6,columnspan=10)

            e1 = tk.Entry(master, textvariable=self.movie_t)
            e2 = tk.Entry(master, textvariable=self.no_usr_rev)
            e3 = tk.Entry(master, textvariable=self.budget)
            e4 = tk.Entry(master, textvariable=self.no_critic_reviews)
            e5 = tk.Entry(master, textvariable=self.fb_likes)
            e6 = tk.Entry(master, textvariable=self.usr_votes)
            e7 = tk.Entry(master, textvariable=self.duration)

            e1.grid(row=0, column=11)
            e2.grid(row=1, column=11)
            e3.grid(row=2, column=11)
            e4.grid(row=3, column=11)
            e5.grid(row=4, column=11)
            e6.grid(row=5, column=11)
            e7.grid(row=6, column=11)

            btn = tk.Button(master,text='Predict',command=self.process_form)
            btn.grid(row=8,column=3,columnspan=5, sticky=tk.W + tk.E)

    def process_form(self):

        a = self.no_usr_rev.get()
        b = self.budget.get()
        c = self.no_critic_reviews.get()
        d = self.fb_likes.get()
        e = self.usr_votes.get()
        f = self.duration.get()

        if a =="" or b =="" or c =="" or d =="" or e =="" or f =="":
            messagebox.showerror("Null Values","Please fill in the empty spaces")
        df = pd.DataFrame([[int(a),float(b),int(c),int(d),int(e),int(f)]],
        columns=['num_user_for_reviews', 'budget','num_critic_for_reviews','movie_facebook_likes',
        'num_voted_users','duration'])

        result = 'The movie '+ self.movie_t.get() + ' is a part of \''+self.tree.predict(df)
        result += '\'. Refer to the classification report for more details'
        messagebox.showinfo("Prediction of the movie",result)


    def process_data_set(self):
        
        #choosing features for decision tree
        columns = [ 'movie_title','num_user_for_reviews', 'budget'
                    , 'num_critic_for_reviews','movie_facebook_likes','num_voted_users','duration']
        
        df = self.df[columns]
        df = df.apply(self.assign_target, axis=1)
        df.drop(labels = ['movie_title'], axis = 1, inplace = True)

        #creating training and test sets
        splitSet = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=0)

        for train_index, test_index in splitSet.split(df, df['cluster']):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        Y_train = train_set.cluster
        X_train = train_set[train_set.columns.drop('cluster')]
        Y_test = test_set.cluster
        X_test = test_set[test_set.columns.drop('cluster')]

        #Creating decision tree 
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        self.tree = decision_tree
        predictions = decision_tree.predict(X_test)

        output = 'Score of the decision tree='+str(decision_tree.score(X_test, Y_test))+('\n')

        output = output+'\nDecision Tree Confusion Matrix\n\n'+str(confusion_matrix(Y_test,predictions))+('\n')
       
        output = output+'\nDecision Tree Classification Report\n\n'+str(classification_report(Y_test,predictions))+('\n')

        #Applying random forest classifier
        rfc = RandomForestClassifier(n_estimators=2000)
        rfc.fit(X_train, Y_train)
        output = output+('Random Forest Statistics\n')

        rfc_pred = rfc.predict(X_test)
        output = output+'\nRandom Forest Confusion Matrix\n\n'+str(confusion_matrix(Y_test,rfc_pred))+('\n')
        
        output = output+'\nRandom Forest Classification Report\n\n'+str(classification_report(Y_test,rfc_pred))+('\n')
       
        print(output)
        self.clicked +=1
        self.results = output


if __name__ == '__main__':
    

    columns = ['movie_title','num_user_for_reviews', 'budget', 'num_critic_for_reviews','movie_facebook_likes',
    'num_voted_users','duration','gross', 'imdb_score']
    
    #loading dataset
    df = pd.read_csv('movie_metadata.csv').dropna(axis=0).reset_index(drop=True)

    dataset = df[['gross', 'imdb_score', 'movie_title']]
    dataset = dataset.values.tolist()

    clusters = kmd.kMedoids(dataset, 4, np.inf, 0)
    
    root = tk.Tk()
    app = MovieApp(root, df[columns], clusters)
    #
    
    root.mainloop()
