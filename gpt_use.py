# -*- coding: latin -*-

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Charger le modèle et le tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt_regis")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Ajouter le token de padding
tokenizer.pad_token = tokenizer.eos_token  # Utiliser le token EOS comme token de padding

# Exemple de texte à générer
input_text = "it appears"

# Tokeniser l'entrée
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Génération avec des paramètres ajustés
generated_output = model.generate(
    **inputs,
    max_length=200,
    #num_beams=5,  # Utilisation de beam search pour une génération plus stable
    no_repeat_ngram_size=2,  # Pour éviter les répétitions
    top_k=50,
    top_p=0.95,  # Utilisation de nucleus sampling
    temperature=0.95,  # Contrôle la "créativité" du texte
    do_sample=True,  # Activer la génération échantillonnée
    pad_token_id=tokenizer.eos_token_id  # Définir explicitement le pad_token_id
)

# Décoder et afficher le texte généré
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(generated_text)
