all: Abgabe.pdf
	
Abgabe.pdf: Bilder/* Tabellen/* content/*.tex ..\LaTeX-Schreibsatz/header.tex lit.bib main.tex build/* | build
	mv main.tex build/main.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	biber build/main.bcf
	lualatex --interaction=nonstopmode --halt-on-error build/main.tex
	lualatex --interaction=nonstopmode --halt-on-error build/main.tex
	mv build/main.tex main.tex 
	mv build/main.pdf Abgabe.pdf 
build:
	mkdir -p build

clean:
	rm -rf build
