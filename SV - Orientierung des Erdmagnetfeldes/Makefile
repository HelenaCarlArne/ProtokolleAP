all: Abgabe.pdf
	
Abgabe.pdf: Bilder/* Tabellen/* content/*.tex ../Allgemein/header.tex main.tex | build
	TEXINPUTS=build:.: lualatex --output-directory=build --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf
	BIBINPUTS=build:. biber build/main.bcf
	TEXINPUTS=build:.: lualatex --output-directory=build --halt-on-error main.tex
	TEXINPUTS=build:.: lualatex --output-directory=build --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf
	rm -rf build/main.*

build:
	mkdir -p build

clean:
	rm -rf build
