from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd
    from tkinter import ttk
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    

    root = tk.Tk()

    root.geometry("800x850+250+5")
    root.title("Crop Prediction")
    root.configure(background="#7D3C98")
    
    
    
    Nitrogen_Level= tk.IntVar()
    Phosphorus_Level= tk.IntVar()
    Potassium_Level = tk.IntVar()
    temperature = tk.IntVar()
    humidity= tk.IntVar()
    ph = tk.IntVar()
    rainfall =  tk.IntVar()
    
    
    
    #===================================================================================================================



    def Detect():
        e1=Nitrogen_Level.get()
        print(e1)
        e2=Phosphorus_Level.get()
        print(e2)
        e3=Potassium_Level.get()
        print(e3)
        e4=temperature.get()
        print(e4)
        e5=humidity.get()
        print(e5)
        e6=ph.get()
        print(e6)
        e7=rainfall.get()
        print(e7)
        
    
        
        #########################################################################################
        
        from joblib import dump , load
        a1=load('D:/crop prediction svm/new.joblib')
        v= a1.predict([[e1, e2, e3, e4, e5, e6, e7]])
        print(v)
    
        if v[0]=='rice':
            print("rice")
            yes = tk.Label(root,text="rice",background="red",foreground="white",font=('times', 20, ' bold '),width=15)
            yes.place(x=500,y=500)
           
                   
        elif v[0]=='maize':
            print("maize")
            no = tk.Label(root, text="maize", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=500, y=500)
        
        elif v[0]=='chickpea':
            print("chickpea")
            no = tk.Label(root, text="chickpea", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=500, y=500)
            
        elif v[0]=='kidneybeans':
            print("kidneybeans")
            no = tk.Label(root, text="kidneybeans", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=500, y=500)
           
        elif v[0]=='pigeonpeas':
            print("pigeonpeas")
            no = tk.Label(root,text="pigeonpeas",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
            no.place(x=500,y=500)
            
        elif v[0]=='mothbeans':
                print("mothbeans")
                yes = tk.Label(root,text="mothbeans",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
                yes.place(x=500,y=500)
           
        elif v[0]=='mungbean':
            print("mungbean")
            yes = tk.Label(root,text="mungbean",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
            yes.place(x=500,y=500)
           
        elif v[0]=='blackgram':
            print("blackgram")
            yes = tk.Label(root,text="blackgram",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
            yes.place(x=500,y=500)
           
        # if v[0]==8:
        #    print("lentil")
        #    yes = tk.Label(root,text="lentil",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==9:
        #    print("pomegranate")
        #    yes = tk.Label(root,text="pomegranate",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==10:
        #    print("banana")
        #    yes = tk.Label(root,text="banana",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==11:
        #     print("mango")
        #     yes = tk.Label(root,text="mango",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #     yes.place(x=600,y=600)
            
        # if v[0]==12:
        #    print("grapes")
        #    yes = tk.Label(root,text="grapes",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==13:
        #    print("watermelon")
        #    yes = tk.Label(root,text="watermelon",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==14:
        #    print("muskmelon")
        #    yes = tk.Label(root,text="muskmelon",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==15:
        #    print("apple")
        #    yes = tk.Label(root,text="apple",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==16:
        #    print("orange")
        #    yes = tk.Label(root,text="orange",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==17:
        #     print("papaya")
        #     yes = tk.Label(root,text="papaya",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #     yes.place(x=600,y=600)
            
        # if v[0]==18:
        #    print("coconut")
        #    yes = tk.Label(root,text="coconut",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
           
        # if v[0]==19:
        #     print("cotton")
        #     yes = tk.Label(root,text="cotton",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #     yes.place(x=600,y=600)
            
        # if v[0]==20:
        #    print("jute")
        #    yes = tk.Label(root,text="jute",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)

        # if v[0]==21:
        #    print("coffee")
        #    yes = tk.Label(root,text="coffee",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
        #    yes.place(x=600,y=600)
       
       
        

    l1=tk.Label(root,text="Nitrogen Level",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l1.place(x=5,y=30)
    Nitrogen_Level=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Nitrogen_Level)
    Nitrogen_Level.place(x=500,y=30)

    l2=tk.Label(root,text="Phosphorus Level",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l2.place(x=5,y=90)
    Phosphorus_Level=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Phosphorus_Level)
    Phosphorus_Level.place(x=500,y=90)

    l3=tk.Label(root,text="Potassium Level",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l3.place(x=5,y=150)
    Potassium_Level=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Potassium_Level)
    Potassium_Level.place(x=500,y=150)
    
    
    l4=tk.Label(root,text="temperature",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l4.place(x=5,y=210)
    temperature=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=temperature)
    temperature.place(x=500,y=210)

    l5=tk.Label(root,text="humidity",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l5.place(x=5,y=270)
    humidity=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=humidity)
    humidity.place(x=500,y=270)

    l6=tk.Label(root,text="ph",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l6.place(x=5,y=330)
    ph=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=ph)
    ph.place(x=500,y=330)
    
    
    l7=tk.Label(root,text="rainfall",background="#D1F2EB",font=('times', 20, ' bold '),width=25)
    l7.place(x=5,y=390)
    rainfall=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=rainfall)
    rainfall.place(x=500,y=390)
    
   
   
    
    button1 = tk.Button(root, foreground="white", background="#283747",text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=300,y=600)


    root.mainloop()

Train()