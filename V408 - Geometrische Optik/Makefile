all: Abgabe.pdf
	
Abgabe.pdf: Makefile Bilder/* Tabellen/* content/*.tex ../Allgemein/header.tex main.tex build/* | build
	max_print_line=1048576 TEXINPUTS=build:.: lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf
	BIBINPUTS=build:. biber build/main.bcf
	max_print_line=1048576 TEXINPUTS=build:.: lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	max_print_line=1048576 TEXINPUTS=build:.: lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	mv build/main.pdf Abgabe.pdf
build:
	mkdir -p build

clean:
	rm -rf build
