#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pickle

#a mettre dans le dossier
import network as rn
import Tagger2 as tg



parser=argparse.ArgumentParser()


parser.add_argument('-w2id', '--word2id', help="dictionnaire wor2id")
parser.add_argument('-t2id', '--tag2id',  help="dictionnaire tag2id")
parser.add_argument('-l', '--load_model', help="load un model en fichier pkl pour le tester sur un texte dans le terminal")

args = parser.parse_args()

def inverse_dico(dic):
  
    return {v : k for k,v in dic.items()}
    

def vectorizer2(phrase, word2id):

    vecteurs = []                  
    
    for i in range(len(phrase)):                               
    
        mot = phrase[i]
    
        #
        if i==0:
            x = np.array([tg.getId("DebutBegin",word2id),tg.getId(mot, word2id),tg.getId(phrase[i+1],word2id)])
       
        #
        elif i==len(phrase)-1:
            x = np.array([tg.getId(phrase[i-1],word2id),tg.getId(mot, word2id),tg.getId("FinEnd",word2id)])
     
        #
        else:
            x = np.array([tg.getId(phrase[i-1],word2id),tg.getId(mot, word2id),tg.getId(phrase[i+1],word2id)])
                
        vecteurs.append(x)
    

    return vecteurs


def application(modele, word2id, tag2id):

      """ fonction qui va tagger un message entree dans le terminal.
      Necessite un modele, un dico xord2id et un dico tag2id
      """


      message = input("Entrer un texte à analyser, en espacant les ponctuations:\n")
      
      id2tag = inverse_dico(tag2id)
  
      resultat=[]
      
      
      phrase = vectorizer2(message.split(' '), word2id)
          
      
      for x in phrase:
          prediction = modele.forward(x)
              
          resultat.append(id2tag[np.argmax(prediction)])
          
      print(list(zip(message.split(' '), resultat)))
      return
              
              
file_model = open(args.load_model, 'rb') 
tagger = pickle.load(file_model)
file_model.close()
tagger = rn.fromModel(tagger)

file_w2id = open(args.word2id, 'rb') 
word2id = pickle.load(file_w2id)
file_w2id.close()


file_t2id = open(args.tag2id, 'rb') 
tag2id = pickle.load(file_t2id)
file_t2id.close()



application(tagger, word2id, tag2id)
