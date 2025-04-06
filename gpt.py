# -*- coding: latin -*-

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling
from dataset import get_tokenized_dataset, get_tokenized_dataset_perso, get_tokenized_dataset_regis  # Assure-toi que cette fonction retourne un dataset
from tqdm import tqdm
import time

if __name__ == '__main__':
    # Verification du peripherique (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modele GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)  # Deplacer le modele sur le peripherique choisi

    # Charger le tokenizer GPT-2
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Ajouter le token de padding
    tokenizer.pad_token = tokenizer.eos_token  # Utiliser le token EOS comme token de padding

    # Charger le dataset tokenise
    dataset = get_tokenized_dataset_regis()

    # Vérifier les colonnes disponibles dans le dataset
    print(f"Colonnes disponibles dans le dataset : {dataset.column_names}")

    # Sélectionner le split 'train' (si c'est ce que vous avez dans votre dataset)
    train_dataset = dataset  # Si vous n'avez pas plusieurs splits, vous utilisez directement 'dataset'


    # Verifier si les splits sont correctement definis
    #print(f"Splits disponibles dans le dataset : {dataset.keys()}")

    # Selectionner le split 'train'
    #train_dataset = dataset['train']

    # Nombre d'exemples dans le dataset
    num_examples = len(train_dataset)  

    # Taille du batch
    batch_size = 2

    # Calcul du nombre de batches
    num_batches_per_epoch = num_examples // batch_size  

    print(f"Nombre de batches par epoque : {num_batches_per_epoch}")

    # Preparer le DataCollator pour la modelisation de langage
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # False pour GPT, car il ne s'agit pas de Masked Language Model (MLM)
    )

    # Preparer le DataLoader pour l'entrainement
    train_loader = DataLoader(
        train_dataset,  # Utilisation du split 'train'
        batch_size=2,  # Ajuste selon la RAM disponible
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4, pin_memory=True
    )

    # Definir l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    elapsed_time = 0
    start_time = time.time()  
    for epoch in range(100):  # Nombre d'epochs ajustable
        
        print("start epoch " + str(epoch + 1))
        model.train()  
        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=True)

        for batch in loop:
            elapsed_time = time.time() - start_time  # 

            
            if elapsed_time > 35 * 60:  # 20 minutes en secondes
                print("20 minutes d'entrainement atteintes, arret du processus.")
                break

            #loop.update(1)
           
            inputs = batch["input_ids"].to(device)

  
            outputs = model(inputs, labels=inputs)

            # Calculer la perte
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

            # Afficher la perte
            #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            loop.set_postfix(loss=loss.item())
    
        if (time.time() - start_time) > 35 * 60:
            break

    # Sauvegarder le modele
    model.save_pretrained("./gpt_regis")
