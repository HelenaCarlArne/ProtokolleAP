all: Abgabe.pdf
	
Abgabe.pdf: Bilder/* Tabellen/* content/*.tex ../Allgemein/header.tex main.tex build/* | build
	cp main.tex build/main.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf 
	biber build/main.bcf
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error build/main.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error build/main.tex
	mv build/main.pdf Abgabe.pdf 
build:
	mkdir -p build

clean:
	rm -rf build
