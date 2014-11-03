all: Abgabe.pdf
	
Abgabe.pdf: *.tex Bilder/* Tabellen/* | build
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error Abgabe.tex
	biber build/Abgabe.bcf
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error Abgabe.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error Abgabe.tex
	mv build/Abgabe.pdf Abgabe.pdf
build:
	mkdir -p build

clean:
	rm -rf build
