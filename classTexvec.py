import re
import os
import copy
import math
class TextVect:
    def __init__(self) -> None:
        pass


    def tokenize(text)->str:
        tok_gram=re.compile(r"""
        (?:etc.|p.ex.|cf.|M.)|
        \w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là)))|
        [\w\-]+'?| . """,re.X)
        return tok_gram.findall(text)
    
    #on prend une liste de tokens en entrée et sort une dictionnaire contenant de token "key" et leur fréquence "value".
    def vectorise(tokens:list)->dict:
        token_freq={}  
        for token in tokens:  
            if token not in token_freq:  
                token_freq[token]=0 
            token_freq[token]+=1 
        return token_freq

    @staticmethod
    def read_text(filename:str)->list:
        try:
            print(filename)
            input_file=open(filename,mode="r",encoding="utf-8")
        except Exception as error:
            print("Impossible de le lire")
        tokens=[]
        vectors_set=[]
        vector_file={}
        #on parcours chaque ligne du fichier, et utilise les méthodes de tokenize et de vectorise 
        #pour sortir une liste contenant de dictionnaire avec les clés de "label" et "vectors".
        for l in input_file:
            l=l.strip()
            token = TextVect.tokenize(l)  
            tokens.extend(token) 
            vector=TextVect.vectorise(tokens)
            vectors_set.append(vector)
        input_file.close()
        vector_file["label"]=""
        vector_file["vectors"]=vectors_set
        return [vector_file]


    def doc2vec(filename:str)->list:
        vectors_final=[]
        #on vérifie si c'est un dossier ou pas
        if os.path.isdir(filename):
            files_exte=os.listdir(filename)
            print("on trouve "+str(len(files_exte))+ " dossiers dans "+ filename)

            #on parcourt chauque dossier dans le dossier extérieur
            for fls in files_exte:
                path=filename+"/"+fls
                if os.path.isdir(path):
                    files_inte=os.listdir(path)
                    #print("on trouve "+str(len(files_inte))+ " files dans "+ path)
                    vectors_set=[]
                    #on parcourt chaque fichier dans chaque dossier intérieur
                    #on parcours chaque ligne du fichier, et utilise les méthodes de tokenize et de vectorise pour sortir une liste contenant de dictionnaire avec les clés de "label" et "vectors".
                    for file in files_inte:
                        tokens=[]
                        try:
                            input_file=open(os.path.join(path,file),'r',encoding='utf-8')
                        except Exception as error:
                            print("Impossible de lire "+file)
                        for l in input_file:
                            l=l.strip()
                            token = TextVect.tokenize(l)  
                            tokens.extend(token) 
                        vector=TextVect.vectorise(tokens)
                        vectors_set.append(vector)
                        input_file.close()
                    vectors_final.append({"label":fls,"vectors":vectors_set})
        return vectors_final
    

    def filtrer(data:list,stopwords:str,hapax:bool)->list:
        f_stop=open(stopwords,'r',encoding='utf-8')
        content_stop=f_stop.read().split("/n")
        vector_final=[]
        vector_set=[]
        #on parcourt chaque vecteur et supprime les mots dans la liste de stopwords et ne compte que les mots dont la fréquence est supérieure à 1.
        for element in data:
            vector_terminal={}
            vector_set=[]
            vectors=element["vectors"]
            for vector in vectors:
                vector_inte={}
                for tk in vector:
                    if tk not in content_stop and (not hapax or vector[tk]>1):
                        vector_inte[tk]=vector[tk]
                vector_set.append(vector_inte)
            vector_terminal["label"]=element["label"]
            vector_terminal["vectors"]=vector_set
            vector_final.append(vector_terminal)
        #print(vector_final)
        
        return vector_final
    

    def tf_idf (data:list)->list:
        data_new=copy.deepcopy(data)
        mots=set()
        #on parcourt tous les tokens et les met dans un set
        for dic in data:
            for vect in dic["vectors"]:
                for tk in vect:
                    mots.add(tk)

        #on crée une dictionnaire contenant tous les tokens comme clé et les fréquences correspondantes comme valeur.
        freq_doc={}
        for m in mots:
            for dic in data:
                for vect in dic["vectors"]:
                    if m in vect:
                        if m not in freq_doc:
                            freq_doc[m]=0
                        freq_doc[m]+=1

        #on calcule les valeur de tf-idf et les met comme les valeurs nouvelles des clés dans cette dictionnaire
        for dic in data_new:
            for vect in dic["vectors"]:
                for token in vect:
                    vect[token] = (vect[token] / len(mots)) * math.log(1 + freq_doc[token])

        return data_new
    
    
    @classmethod
    #traiter les textes de train.
    def traintovect(cls):
        vecteur_resultat=cls.doc2vec("corpus")
        vecteur_filtrer=cls.filtrer(vecteur_resultat,"data/stopliste.txt",False)
        donnes=cls.tf_idf(vecteur_filtrer)
        return donnes
    
    @classmethod
    #traiter le texte de test en str vers vecteur en dict.
    def testtovect(cls,filename):
        contenu=TextVect.read_text(filename)
        contenu_filtre=TextVect.filtrer(contenu,"data/stopliste.txt",False)
        vector=TextVect.tf_idf(contenu_filtre)
        return vector[0]["vectors"][0]
