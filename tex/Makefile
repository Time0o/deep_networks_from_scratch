TEX_DIR=tex
OUT_DIR=out
PDF_DIR=pdf

all: $(patsubst $(TEX_DIR)/%.tex, $(PDF_DIR)/%.pdf, $(wildcard $(TEX_DIR)/*.tex))

$(PDF_DIR)/%.pdf: $(TEX_DIR)/%.tex
	@pdflatex --shell-escape $<
	@pdflatex --shell-escape $<
	mv *.aux *.log $(OUT_DIR)
	mv $(notdir $@) $(PDF_DIR)

.PHONY: clean

clean:
	rm -f $(OUT_DIR)/* $(PDF_DIR)/*
