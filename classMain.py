from classTexvec import TextVect
from knnClasses import KNNClasses
class Main:
    if __name__=="__main__":
        object1 = TextVect()
        donnee=TextVect.traintovect()
        object_Knn=KNNClasses("test",donnee)
        
        while True:
            input_method = input("Choisissez la methode: ")
            match input_method:
                case "addclass":
                    label = input("Entrez la label de class: ")
                    vectors = input("Entrez le vecteur que vous voulez ajouter (les séparer par espace): ")
                    if type(vectors)!=dict:
                        vectors=TextVect.testtovect(vectors)
                    resultat=object_Knn.add_class(label, vectors)
                    print(resultat) 

                case "addvector":
                    label = input("Entrez la label de class dans laquelle vous voulez ajouter le vecteur correspondante: ")
                    vector = input("Entrez le vecteur que vous voulez ajouter (les séparer par espace): ")
                    if type(vector)!=dict:
                        vector=TextVect.testtovect(vector)
                    resul=object_Knn.add_vector(label, vector)
                    print(resul) 

                case "delclass":
                    label = input("Entrez la label de class que vous voulez supprimer: ")
                    resul=object_Knn.del_class(label)
                    print(resul)  

                case "savejson":
                    filename = input("Entrez le nom de fichier sur lequel vous voulez sauvegarder: ")
                    object_Knn.save_as_json(filename) 

                case "loadjson":
                    filename = input("Entrez le nom de fichier vous voulez charger: ")
                    content=object_Knn.load_as_json(filename)
                    print(content)

                case "classify":
                    vector = input("Entrez le vecteur que vous voulez classifier (les séparer par espace): ")
                    if type(vector)!=dict:
                        vector=TextVect.testtovect(vector)
                    k = int(input("Entrez la valeur k: "))
                    sim_func = input("quel méthode vous voulez utiliser pour calculer la distence entre deux vecteurs ?\n \
                                    répondez cos pour la méthode consinus,\n \
                                    répondez eucli pour la méthode de distance euclidean: ")
                    if sim_func == "cos":
                        sim_func=KNNClasses.cosinus
                    elif sim_func == "eucli":
                        sim_func=KNNClasses.eucliDist
                    else:
                        sim_func=KNNClasses.cosinus
                    result = object_Knn.classify(vector, k,sim_func)
                    print(result) 

                case "print":
                    object_Knn.printclass()

                case "exit":
                    break
                case _:

                    print("Erruer! Essayez encore une fois. Veuillez entrer les reponses concrètes.")
