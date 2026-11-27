"""Connexion à Spotify et diagnostic (à lancer une fois avant python main.py --player spotify).

    python scripts/spotify_setup.py

1. ouvre le navigateur pour autoriser Maestro sur votre compte (une seule fois, jeton gardé dans models/) ;
2. vérifie le compte Premium ;
3. liste vos playlists (et celles dont Maestro peut lire les morceaux) et vos appareils Spotify.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maestro import config  # noqa: E402
from maestro.spotify_player import SpotifyPlayer, friendly_error  # noqa: E402


def main():
    sp = SpotifyPlayer._connect()
    try:
        me = sp.current_user()
    except Exception as exc:
        sys.exit(f"Connexion refusée : {friendly_error(exc)}\n"
                 f"Vérifiez le Client ID / Secret et que l'URI de redirection du dashboard est exactement\n"
                 f"  {config.SPOTIFY_REDIRECT_URI}")
    product = me.get("product", "?")
    print(f"Connecté : {me.get('display_name') or me.get('id')}  |  abonnement : {product}")
    if product not in ("premium", "?"):
        print("  ATTENTION : le contrôle de la lecture exige Spotify Premium.")

    print("\nVos playlists (★ = morceaux lisibles par Maestro : playlist à vous ou collaborative) :")
    page = sp.current_user_playlists(limit=50)
    for p in page.get("items", []):
        if not p:
            continue
        mine = (p.get("owner") or {}).get("id") == me.get("id") or p.get("collaborative")
        print(f"  {'★' if mine else ' '} {p['name']}")
    print("  -> mettez le nom choisi dans .env :  SPOTIFY_PLAYLIST=<nom>")

    devices = sp.devices().get("devices", [])
    print("\nAppareils Spotify :")
    for d in devices:
        print(f"  - {d['name']} ({d['type']}){'  [actif]' if d.get('is_active') else ''}")
    if not devices:
        print("  aucun : ouvrez l'application Spotify (PC ou téléphone) puis relancez ce script.")
    print("\nPrêt : python main.py --player spotify   (ou PLAYER=spotify dans .env)")


if __name__ == "__main__":
    main()
