import math
import json

class KNNClasses:
    def __init__(self,description,data):
        self.description=description
        self.data=data

    #Cette fonction a pour objectif d'afficher la data en entrée
    def printclass(self):
        print(self.data)
    
    def add_class(self,label:str,vectors:list):
        self.data.append({
            "label":label,
            "vectors":vectors
        }
        )
        return self.data

    def add_vector(self, label:str, vector:list):
        for i in range(len(self.data)):
            if self.data[i]["label"] == label:
                self.data[i]["vectors"].append(vector)
            
        return self.data
    
    def del_class(self,label:str):
        for i in range(len(self.data)):
            if self.data[i]["label"]==label:
                del self.data[i]

                return self.data
            else:
                print("impossble de trouver ce label et ses vectors")

    def save_as_json(self,filename:str):
        for each in self.data:
            with open(filename+".json",mode="w",encoding="utf-8") as f:
                json.dump(each,f)  

            return True
        
    def load_as_json(self,filename:str):
        with open(filename,mode="r",encoding="utf-8") as f:
            f_content=json.load(f)
            #print(f_content)
        return f_content

    @staticmethod
    #cette fonction sort les valeurs des clés et les met dans une liste utilisant une boucle.
    def vect_to_value_list(vect:dict)->list:
        vect_value=[]
        for key in vect.keys():
            vect_value.append(vect[key])

        return vect_value
    
    @staticmethod
    #calculer la distance euclidienne entre deux vecteurs.
    def eucliDist(vect1:dict,vect2:dict)->float:
        vect1_val=KNNClasses.vect_to_value_list(vect1)
        vect2_val=KNNClasses.vect_to_value_list(vect2)
        eucliDist=math.sqrt(sum([(m - n)**2 for (m,n) in zip(vect1_val,vect2_val)]))
        return eucliDist

    @staticmethod
    #on calcule la valeur 'dot' : la somme des produits des éléments de même index "clé" dans les deux vecteurs.
    def dotvalue(vector1:dict,vector2:dict)->float:
            liste_scalaire=[]
        #pour trouver les éléments ayant de même clé dans les deux vecteurs.
        for key in vector1:
            if key in vector2:
                liste_scalaire.append(vector1[key]*vector2[key])
        produit_scalaire=sum(liste_scalaire)
        return produit_scalaire
    
    @staticmethod
    #cette fonction a pour objectif de normaliser le vecteur.
    def normalise(vector:dict)->float:
        norme_carre=0
        for key in vector:
            norme_carre+=vector[key]**2.0
        norme=math.sqrt(norme_carre)
        return norme
    
    @staticmethod
    #cette fonction calcule la similarité de consinus de deux vecteurs.
    def cosinus(vector1,vector2)->float:
        norme1=KNNClasses.normalise(vector1)
        norme2=KNNClasses.normalise(vector2)
        scal=KNNClasses.dotvalue(vector1,vector2)
        cosinus=(scal/(norme1*norme2))
        
        return cosinus

        
    def classify(self,vector:dict,k:int, sim_func)->list:
        dict_sim={}
        
        #on parcourt chauque vecteur utilisant deux boucles.
        for each in self.data:
            value=[]
            for vect in each["vectors"]:
                #calculer la similarité entre le vectuer "test" et chaque vecteur de data.
                value.append(sim_func(vect,vector))
            dict_sim[each["label"]]=value

        #on sort tous les valeurs de similarité de dict_sim utilisant deux boucles
        #et les met dans une nouvelle liste.
        list_sim=[]
        for sim in dict_sim.values():
            for simval in sim:
                list_sim.append(simval)
        
        list_simre=sorted(list_sim,reverse=True)
        list_simnew=[]
        #on parcourt les k premiers chiffres de liste décroissante contenant les similarités
        for i in range(k):
            dict_simnew={}
            #on parcourt dict_sim pour trouver les labels des k permiers chiffres
            for j in dict_sim:
                if list_simre[i] in dict_sim[j]: 
                    dict_simnew["label"]=j 
                    #on met la moyenne des similarités obtenues sur les vecteurs d'une classe comme la valeur de "sim"
                    dict_simnew["sim"]=float(sum(dict_sim[j])/len(dict_sim[j]))
                    list_simnew.append(dict_simnew)
                    break
       
        for each in list_simnew:
            print ("label: "+each["label"]+", k= "+str(k)+", sim: "+str(each["sim"]))
        return list_simnew
