$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -shell-escape %O %S';
$out_dir = 'out';
$aux_dir = 'out';
$pdf_mode = 1;
$bibtex = "bibtex %O %B";
@BIBINPUTS = ( ".", "../" );
@default_files = ('report.tex')