#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from collections import defaultdict
import numpy as np
from math import exp
from time import time
from random import shuffle, sample
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pickle

# A mettre dans le dossier
import network as rn





def make_dictionnaries(fichier):
  
    """ Création de dictionnaires contenant les occurences, les mots et les id des mots
	
	Parametres:
	fichier : un fichier de format conll
	
	Returns:
	occurences: dictionnaire contenant les occurences de chaque mots dans le texte passé en arglimiteument
	word2id: dictionnaire contenant les mots et leurs id correspondant
	tag2id: ditionnnaire contenant les ids et leurs mots correspondant
    
    """    
    
    occurences = defaultdict(int)             #

    word2id = {}                              #
    id_word=0                                 #

    tag2id = {}                               #
    id_tag=0                                  #

    
    #
    with open(fichier, 'r', encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#") and line!="\n":
                [lidx,mot,lemme,cat]=line.strip('\n').strip('\r').split('\t')[:4]
                if "-" not in lidx and "." not in lidx :
                    occurences[mot]+=1
                    if cat not in tag2id:
                        tag2id[cat]=id_tag
                        id_tag += 1
                                        
    #création de word2id
    for word, occ in occurences.items():
        if occ > 1:
            word2id[word] = id_word
            id_word += 1
        
    #
    word2id["DebutBegin"]=id_word
    id_word +=1
    word2id["FinEnd"]=id_word
    id_word +=1
    word2id["UnknowInconnu"]=id_word
    id_word +=1

    print(100*(len(word2id)-3)/len(occurences), "% des mots apparaissant plus d'une fois dans",fichier,"soit un vocabulaire de taille", len(word2id)-3,"+1 pour inconnu")

    return occurences, word2id, tag2id
  
def getId(word, word2id):
  
    """ Renvoie l'id du mot entré en paramètre
	
	Parametres:
	word:le mot dont on veut connaitre l'id
	word2id: le dictionnaire correspondant à mot -> id
        
	Returns:
	word2id[word]: l'id correspodant au mot
    """
  
    #
    if word not in word2id:
        return word2id["UnknowInconnu"]
        
    #
    else:
        return word2id[word]
          
          
def extract_word2vec(fichier, words_indices):
  
    """ Prend un fichier contenant des word2vec et extrait ces word2vec
      
	  Parametres:
	  fichier: le fichier contenant les word2vec
	  words_indices: dictionnaire contenant les indices des mots 
	  
	  Returns:
	  word2vec: les word2vecs
	  nb_feats: le nombre de features
    """ 
    
    word2vec={}                   #
    
    #
    with open(fichier,"r",encoding="utf-8") as file:
        for line in file:
            line = line.replace(" \n","").split(" ")
            # Lecture des informations du fichier
            # nombre de mots presents et nombre de features
            if len(line)==2 :
                nb_words=int(line[0])
                nb_feats=int(line[1])
                              
            #
            else:
                if line[0] in words_indices:
                    word, vec = line[0],np.array(line[1:])
                    word2vec[word]=vec

    print("{} embbedings de taille {} pertinent parmi les {} du fichier".format(len(word2vec), nb_feats, nb_words))

    return word2vec, nb_feats
      
      
        
def build_embbedings(word2id, nb_feats, word2vec=None ):
    """ Crée une matrice d'embeddings de taille len(dico)*nb_feats
	
	Parametres:
	word2id: dictionnaire correspondant à mot -> id
	nb_feats: le nombre de features
	word2vec: un vecteur représentant le mot initialisé vide
	
	Returns:
	matrix_embbedings: une matrice d'embbedings
    
    """
   
    matrix_embbedings = np.random.rand(len(word2id),nb_feats)         # Initialisation random des embeddings
      
    #
    if word2vec!=None:
        for word,indice in word2id.items():
            if word in word2vec:
                matrix_embbedings[indice,:]=np.copy(word2vec[word])  #copie l'embedding du word2vec
                
    return matrix_embbedings
  
    
def lecture_phrase(fichier):
  
    """ Lis un fichier de format conll et stock les phrases
	
	Parametres:
	fichier: Le fichier en format conll
	
	Returns:
	phrases: liste; la totalité des phrases du fichier
        
    """
  
    phrases= []                 # liste contenant les phrases
    phrase = []                 # liste contenant les mots de la phrase en lecture
      
      
    with open(fichier,'r',encoding='utf-8') as file:
        for line in file:
            if not line.startswith("#"):
                if line!="\n":
                    [lidx,mot,lemme,cat]=line.strip('\n').strip('\r').split('\t')[:4]
                    if "-" not in lidx and "." not in lidx :
                        phrase.append([lidx,mot,lemme,cat])
                elif phrase!=[]:
                    phrases.append(phrase)
                    phrase=[]

    print(len(phrases), "phrases extraites de", fichier)
    
    return phrases
      

def vectorizer(phrase, word2id, tag2id):
    """ Renvoie un vecteur d'un mot à partir des données de sa phrase correspondante
	
	Parametres:
	phrase : la phrase actuelle contenant le mot
        word2id: dict; mots vers leurs indices
        tag2id : dict; tags vers leurs indices
	
	Returns:
	une liste de tuple correspondant au vecteur (id mot prec,id mot a tag, id mot suivant) du mot et son de son gold
    
    """

    vecteurs = []                  #
    golds = []                     #

    #
    for i in range(len(phrase)):                               
    
        lidx,mot,lemme,cat = phrase[i]
        y= tag2id[cat]
    
        #
        if i==0:
            x = np.array([getId("DebutBegin",word2id),getId(mot, word2id),getId(phrase[i+1][1],word2id)])
       
        #
        elif i==len(phrase)-1:
            x = np.array([getId(phrase[i-1][1],word2id),getId(mot, word2id),getId("FinEnd",word2id)])
     
        #
        else:
            x = np.array([getId(phrase[i-1][1],word2id),getId(mot, word2id),getId(phrase[i+1][1],word2id)])
    
        vecteurs.append(x)
        golds.append(y)
    

    return list(zip(vecteurs, golds))








def train_tagging( model, optimiseur, data, word2id , tag2id):

    """ Entraine le modèle sur le set de données et renvoie la perte globale
	
	Parametres:
	model: Network; reseau de neurones qui est entrainé.
	data: les données (les fichiers) que l'on donne pour évaluation
	word2id: dictionnaire contenant les mots et leurs id correspondant
	tag2id: ditionnnaire contenant les ids et leurs mots correspondant
	
	Returns:
	perte_global durant l'entrainement
  
    """
    
    NLLLs=[]                                                 #
    
    #
    for phrase in sample(data,len(data)):
        phrase = vectorizer(phrase, word2id, tag2id)
        
        #
        for x,y in sample(phrase,len(phrase)):
            prediction = model.forward(x)                    #
      
            NLLL = rn.NegativeLogLikelihood(prediction, y)      #
            NLLLs.append(NLLL)
            
            #
            if (np.argmax(prediction)!=y):
        
                # gradient de la fonction de perte
                gradient_NLLL=np.zeros(prediction.shape).T     # creation d'un vecteur nul de la taille voulue
                gradient_NLLL[y,0]=-1/(prediction[0,y])        # modification du vecteur à l'indice gold avec la valeur de la derivée de la perte

                #Optimisation pour mise à jour des poids
                optimiseur.step(gradient_NLLL, y, x)
                

    perte_global=sum(NLLLs)/len(NLLLs)                   # nlll global sur l'ensemble
    
    return perte_global
        
def eval_nlll(model, data, word2id, tag2id):

    """Calcul la perte global
	
	Parametres:
	model: Network; reseau de neurones qui est entrainé.
	data: les données (les fichiers) que l'on donne pour évaluation
	word2id: dictionnaire contenant les mots et leurs id correspondant
	tag2id: ditionnnaire contenant les ids et leurs mots correspondant
	
	Returns:
	perte_global : la perte estimée pour notre corpus
    
    """
    NLLLs=[]                                         #
        
    #
    for phrase in data:
        phrase = vectorizer(phrase, word2id, tag2id)
        
        for x,y in phrase:
          
            prediction = model.forward(x)            #
            
            NLLLs.append( rn.NegativeLogLikelihood(prediction, y) )
              
        
    perte_global=sum(NLLLs)/len(NLLLs)              # nlll global sur l'ensemble
            
    return perte_global
      
      
def eval_acc(model, data, word2id, tag2id):

    """ Calcul la précision
	
	Parametres:
	model: Network; reseau de neurones qui est entrainé.
	data: les données (les fichiers) que l'on donne pour calculer la précision
	word2id: dictionnaire contenant les mots et leurs id correspondant
	tag2id: ditionnnaire contenant les ids et leurs mots correspondant
	
	Returns:
	L'accuracy/précision sur tous les mots, et sur les mots inconnus
    
    """
    total=0                                              #
    correct=0                                            #
    unk_total = 0                                        #
    unk_correct = 0                                      #
    
    for phrase in data:
        phrase = vectorizer(phrase, word2id, tag2id)
        
        for x,y in phrase:
            prediction = model.forward(x)                 #
            
            total += 1

            #mot a tag est-il unk
            if x[1]==len(word2id)-1:
                unk_total+=1
                
            #
            if (np.argmax(prediction)==y):
                correct += 1

                if x[1]==len(word2id)-1:
                    unk_correct+=1
                    
    if unk_total==0:
        unk_total=1
        unk_correct=1

    return 100*correct/total, 100*unk_correct/unk_total
      
      
      
      

def early_stop(model, optimiseur, train, dev, word2id, tag2id, epoch_max=50, steps = 1, patience = 5, fichier="graph", suivi= False):
  
    """ Early stop qui retourne le meilleur reseau, inspiré du code existant en ligne
        REF: https://gist.github.com/ryanpeach/9ef833745215499e77a2a92e71f89ce2
        et Algorithm 7.1 dans deep learning book.
  
        Parametres:
        model: Network; reseau de neurones qui est entrainé.
        train: list; Corpus de train.
        dev: list; Corpus de dev.
        steps: int; Le pas entre les evaluations de la perte.
        patience: int; Le nombre limite où la perte augmente consécutivement.
          
        Returns:
        best: dict; le meilleur reseau evalué compressé en dict.
        k: int; Le nombre d'epoch nécesssaire pour arriver à best.
        loss: float; la perte de best sur dev.
        epoches: list; La liste des epoch avec evaluation
        nllls: list; la liste des pertes aux epoch de epoches
    """
  
    start = time()
  
  
    best = model.save()          # Le meilleur model, au début celui en entrée
    i = 0                        # Le nombre d'epoch effectué
    j = 0                        # Le nombre d'evaluations par pas depuis la derniere update de best
    loss = np.inf                # La meilleure perte
    k = i                        # Le nombre d'epoch pour atteindre best 
  
    epoches = []                 # La liste des epoch avec evaluation
    nlll_train = []              # La liste des pertes sur train aux epoch de epoches
    nlll_dev = []                # La liste des pertes sur dev aux epoch de epoches
  
    while j < patience and i < epoch_max:
        # steps trainning
        for _ in range(steps):
            loss_train = train_tagging(model, optimiseur, train, word2id, tag2id)

        # Mise a jour de l'epoch et evaluation de la nouvelle perte
        i += steps
        new_loss = eval_nlll(model, dev, word2id, tag2id)

        if suivi:
            print("epoch", i, "loss sur dev", new_loss, "loss sur train", loss_train)
        
        epoches.append(i)
        nlll_dev.append(new_loss)
        nlll_train.append(loss_train)

        # Si la perte est meilleure, save le model
        if new_loss < loss:
            j = 0
            best = model.save()
            k = i
            loss = new_loss

        # Sinon incremente le nombre d'evaluation effectue
        else:
            j += 1
            
    print("En", time()-start) 
            
    plt.plot(epoches, nlll_dev, label='dev')
    plt.plot(epoches, nlll_train, label='train')
    
    plt.legend()

    if fichier!=None:
        plt.savefig(fichier)
        plt.close()
    else:
        plt.show()
    
    return best, k, loss



if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-nc', '--nb_couches', default= 3, type=int, help = "nombre de couches cachés. Default=1")
    parser.add_argument('-nn', '--nb_neurones', default= 300,type=int, help = "nombre de neurones. Default=50")
    parser.add_argument('-lr', '--learning_rate', default= 0.01, type=float, help = "valeur du learning rate. Default=0.01")
    parser.add_argument('-e', '--epoch', default= 30, type=int, help = "nombre d'epoch max. Default=30")
    parser.add_argument('-p', '--patience', default= 5, type=int, help = "nombre d'attente pour fin du early stop. Default=5")
    parser.add_argument('-train', help = 'Fichier train conll ')
    parser.add_argument('-dev', help = 'Fichier dev conll')
    parser.add_argument('-test', help = 'Fichier test conll')
    parser.add_argument('-vec', '--vec_file', default='', help="Fichier fournissant les word2vecs. Default='' : pas de vecteurs")
    parser.add_argument('-f', '--nb_feats', default=50, type=int, help="Nombre de features pour les embbedings. Default=50")
    parser.add_argument('-gs', '--gridsearch', action="store_true" ,  help="gridsearch pour recherche du meilleur model. Default= False")
    parser.add_argument('-s', '--suivi', action="store_true" ,  help="suivi de l'eraly stop. Default= False")
    parser.add_argument('-sf', '--save_fichier', default='model.pkl', help="Fichier contenant le model au format pkl. Default='model.pkl'")

    parser.add_argument('-lo', '--load', default='' , help="load un model en fichier pkl pour le tester. Default='' :pas de model a charger")
    args = parser.parse_args()
    
    word2occ , word2id , tag2id = make_dictionnaries(args.train)

    output = open("w2id.pkl", 'wb')
    pickle.dump(word2id, output)
    output.close()

    output = open("t2id.pkl", 'wb')
    pickle.dump(tag2id, output)
    output.close()

    
    if args.vec_file != '':
        word2vec , nb_feats = extract_word2vec(args.vec_file , word2id)
        matrix_embbedings=build_embbedings(word2id, nb_feats, word2vec)
        
    else:
        nb_feats = args.nb_feats
        matrix_embbedings=build_embbedings(word2id, nb_feats)
        
        
        
    train = lecture_phrase(args.train)    
    dev = lecture_phrase(args.dev)
    test = lecture_phrase(args.test)
    
    print()
    
    tanh= rn.Tanh()
    
    
    softmax= rn.Softmax()
    
    if args.load!='':

        file_model = open(args.load, 'rb') 
        tagger = pickle.load(file_model)
        file_model.close()


        tagger = rn.fromModel(tagger)
        print("nlll sur test", eval_nlll(tagger, test, word2id, tag2id))

        total, unk = eval_acc(tagger, test, word2id, tag2id)
        print("accuracies sur test: total", total, "unk", unk )
        
    elif args.gridsearch:
        best_model=(None, np.inf)
        
        for nb_couche in [1,2,3]:
            for nb_neurones in [50, 60, 70, 80, 90,100, 200, 300]:
                for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                    fname="cc={}+nbn={}+lr={}".format(nb_couche, nb_neurones, lr).replace(".","_")
                    
                    print(fname)
                    
                    Tagger = rn.Network(3, len(tag2id), tanh, softmax, nb_couche, nb_neurones,
                                        n1=3, nb_embeddings1=len(word2id), nb_feats1=nb_feats)
                    
                    Tagger.setEmbeddings([matrix_embbedings])
                    
                    Opti = rn.SGD(Tagger, lr , embeddings_layers=True)
                    
                    best, k, loss= early_stop(Tagger, Opti, train, dev, word2id, tag2id,
                                              epoch_max= args.epoch , patience = args.patience, fichier=None, suivi=args.suivi)
                    
                    print(best["hyper_parameters"], "learning rate:", lr)
                    print("meilleur perte", loss, "a l'epoch", k)

                    print()
                    
                    if loss < best_model[1]:
                        best_model=(best, loss)
                        
                    del best
                    del k
                    del loss
                    
                        
        TaggerReload = rn.fromModel(best_model[0])
            
        print("nlll sur test", eval_nlll(TaggerReload, test, word2id, tag2id))
        total, unk = eval_acc(TaggerReload, test, word2id, tag2id)
        print("accuracies sur test: total", total, "unk", unk )
        
        #sauvegarde du fichier contenant le model
        output = open(args.save_fichier, 'wb')
        pickle.dump(best_model[0], output)
        output.close()

    else:


        
        Tagger = rn.Network(3, len(tag2id), tanh, softmax, args.nb_couches, args.nb_neurones,
                            n1=3, nb_embeddings1=len(word2id), nb_feats1=nb_feats)
                    
        Tagger.setEmbeddings([matrix_embbedings])
        
        Opti = rn.SGD(Tagger, args.learning_rate , embeddings_layers=True)
        
        best, k, loss = early_stop(Tagger, Opti, train, dev, word2id, tag2id,
                                   epoch_max= args.epoch , patience = args.patience, fichier=None, suivi=args.suivi)

        print("meilleur perte", loss, "a l'epoch", k)

        print("nlll sur test", eval_nlll(Tagger, test, word2id, tag2id))
        total, unk = eval_acc(Tagger, test, word2id, tag2id)
        print("accuracies sur test: total", total, "unk", unk )
        
        #sauvegarde du fichier contenant le model
        output = open(args.save_fichier, 'wb')
        pickle.dump(best, output)
        output.close()

            

            
                        
