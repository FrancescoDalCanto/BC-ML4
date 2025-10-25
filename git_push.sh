#!/bin/bash

# Testo da inserire nel commento
echo "Inserisci il messaggio di commit:"
read commit_message
# Comandi
git add .
git commit -m "$commit_message"
git push

echo "Push completato con successo!"