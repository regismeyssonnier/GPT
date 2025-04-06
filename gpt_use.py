# -*- coding: latin -*-

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Charger le mod�le et le tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt_regis")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Ajouter le token de padding
tokenizer.pad_token = tokenizer.eos_token  # Utiliser le token EOS comme token de padding

# Exemple de texte � g�n�rer
input_text = "it appears"

# Tokeniser l'entr�e
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# G�n�ration avec des param�tres ajust�s
generated_output = model.generate(
    **inputs,
    max_length=200,
    #num_beams=5,  # Utilisation de beam search pour une g�n�ration plus stable
    no_repeat_ngram_size=2,  # Pour �viter les r�p�titions
    top_k=50,
    top_p=0.95,  # Utilisation de nucleus sampling
    temperature=0.95,  # Contr�le la "cr�ativit�" du texte
    do_sample=True,  # Activer la g�n�ration �chantillonn�e
    pad_token_id=tokenizer.eos_token_id  # D�finir explicitement le pad_token_id
)

# D�coder et afficher le texte g�n�r�
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(generated_text)
