all: Abgabe.pdf
	
Abgabe.pdf: Makefile Bilder/* Tabellen/* content/*.tex ../LaTeX-Schreibsatz/header.tex main.tex build/* | build
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	biber build/main.bcf
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf 
build:
	mkdir -p build

clean:
	rm -rf build
