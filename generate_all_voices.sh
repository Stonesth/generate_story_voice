#!/bin/bash

# Liste des speakers
speakers=(
    "VCTK_p225" "VCTK_p226" "VCTK_p227" "VCTK_p228" "VCTK_p229" "VCTK_p230" 
    "VCTK_p231" "VCTK_p232" "VCTK_p233" "VCTK_p234" "VCTK_p236" "VCTK_p237" 
    "VCTK_p238" "VCTK_p239" "VCTK_p240" "VCTK_p241" "VCTK_p243" "VCTK_p244" 
    "VCTK_p245" "VCTK_p246" "VCTK_p247" "VCTK_p248" "VCTK_p249" "VCTK_p250" 
    "VCTK_p251" "VCTK_p252" "VCTK_p253" "VCTK_p254" "VCTK_p255" "VCTK_p256" 
    "VCTK_p257" "VCTK_p258" "VCTK_p259" "VCTK_p260" "VCTK_p261" "VCTK_p262" 
    "VCTK_p263" "VCTK_p264" "VCTK_p265" "VCTK_p266" "VCTK_p267" "VCTK_p268" 
    "VCTK_p269" "VCTK_p270" "VCTK_p271" "VCTK_p272" "VCTK_p273" "VCTK_p274" 
    "VCTK_p275" "VCTK_p276" "VCTK_p277" "VCTK_p278" "VCTK_p279" "VCTK_p280" 
    "VCTK_p281" "VCTK_p282" "VCTK_p283" "VCTK_p284" "VCTK_p285" "VCTK_p286" 
    "VCTK_p287" "VCTK_p288" "VCTK_p292" "VCTK_p293" "VCTK_p294" "VCTK_p295" 
    "VCTK_p297" "VCTK_p298" "VCTK_p299" "VCTK_p300" "VCTK_p301" "VCTK_p302" 
    "VCTK_p303" "VCTK_p304" "VCTK_p305" "VCTK_p306" "VCTK_p307" "VCTK_p308" 
    "VCTK_p310" "VCTK_p311" "VCTK_p312" "VCTK_p313" "VCTK_p314" "VCTK_p316" 
    "VCTK_p317" "VCTK_p318" "VCTK_p323" "VCTK_p326" "VCTK_p329" "VCTK_p330" 
    "VCTK_p333" "VCTK_p334" "VCTK_p335" "VCTK_p336" "VCTK_p339" "VCTK_p340" 
    "VCTK_p341" "VCTK_p343" "VCTK_p345" "VCTK_p347" "VCTK_p351" "VCTK_p360" 
    "VCTK_p361" "VCTK_p362" "VCTK_p363" "VCTK_p364" "VCTK_p374" "VCTK_p376"
)

# Créer le dossier pour tous les speakers s'il n'existe pas
mkdir -p story_output/all_speakers

# Générer l'audio pour chaque speaker
for speaker in "${speakers[@]}"; do
    echo "Génération de l'audio pour le speaker: $speaker"
    python Simple_TTS.py --lang 2 --en-model 3 --text-file text_en.txt --use-cuda --speaker "$speaker"
    # Déplacer le fichier généré dans le sous-dossier all_speakers
    mv "story_output/output_en_vctk_${speaker}.wav" "story_output/all_speakers/"
done

echo "Génération terminée ! Tous les fichiers audio sont dans le dossier story_output/all_speakers/"
