Nach langer und etwas ärgerlicher Arbeit, habe ich es nun geschafft, ein Repository zu erstellen.
Dieses Repository funktioniert so, dass jedes Protokoll einen eigenen Ordner bekommt und stets am Anfang "gepullt" und am Ende "gepusht" wird.

Hier eine kurze Anleitung, liebe Helena, wie ich git verstehe:

1. Erstellen von Repositorys
	A. Erstelle einen Ordner mit dem Namen Deines (zukünftigen) Repositorys.
	B. Öffne den Ordner in der Kommandozeile
	C. Melde bei GitHub das Repos an
	D. Verwende in der Kommandozeile die Befehle, die auf GitHub angezeigt werden:
		(a) touch README.md
		(b) git init
		(c) git add README.md
		(c*)git add {andere Dateien}
		(d) git commit -m "first commit"
		(e) git remote add origin {Adresse von GitHub.com}
		(f) git push -u origin master
	Legende:
	a – Erstellt eine leere Datei, die als erste Datei ins entfernte Repo. geladen wird.
	b – Initialisiert das git-System
	c - Fügt die Dateien zu dem Staging hinzu, ab jetzt kann git darauf Eifluss nehmen. Vor dem "Adden" war das nicht möglich
	d - Veranlasst das Übernehmen der Änderungen zu einem Datencommit, welcher bereit ist, auf den Server gelegt zu werden.
	e - Hiermit wird das Ur-Repository vom Rechner auf den Server geladen. Damit ist git in gewohnter Weise betriebsbereit.
	f - Das ist der erste Push-Befehl. Alle folgenden sind "git push origin master" oder "git push"