#!/bin/bash
echo "🔍 Vérification du fichier .env.local..."
echo ""
echo "Variables Supabase trouvées :"
grep "NEXT_PUBLIC_SUPABASE\|SUPABASE_SERVICE" .env.local | while IFS= read -r line; do
  var_name=$(echo "$line" | cut -d'=' -f1)
  var_value=$(echo "$line" | cut -d'=' -f2-)
  if [ -z "$var_value" ]; then
    echo "  ❌ $var_name : VIDE"
  elif [ $(echo "$var_value" | wc -l) -gt 1 ]; then
    echo "  ⚠️  $var_name : COUPÉE SUR PLUSIEURS LIGNES"
  else
    echo "  ✅ $var_name : OK (${#var_value} caractères)"
  fi
done
