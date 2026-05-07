%
%       HISTORY:
%               okman - April 25, 1996: Created,
%               chlee - April 25, 1997: Modified,
%                       from report.cls + doublespace.sty + sythesis.sty
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin of "report.cls" Modification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[1999/04/30 Kaist PhD Thesis class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass{book}
\renewcommand\figurename{Figure.}
\renewcommand\tablename{Table.}
\newenvironment{MYitemize}{%
    \renewcommand{\labelitemi}{\ding{111}}%
    \renewcommand{\labelitemii}{\ding{43}}%
    \renewcommand{\labelitemiii}{\ding{52}}%
    \renewcommand{\labelitemiv}{\ding{45}}%
    \begin{itemize}}{\end{itemize}}
\newcommand {\bsi} {\begin{MYitemize}}
\newcommand {\esi} {\end{MYitemize}}
\newcommand\fchapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@fchapter\@schapter}
\newcommand{\QED}{\mbox{\rule[0pt]{1.2ex}{1.2ex}}}
\newenvironment{proof}{\noindent\hspace{2em}{\it Proof: }} 
            {\hspace*{\fill}~\QED}

\def\@chapter[#1]#2#3#4{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{protect#1}
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}{#3}{#4}]%
                    \else
                      \@makechapterhead{#2}{#3}{#4}%
                      \@afterheading
                    \fi}

\def\@makechapterhead#1#2#3{%
  \noindent
  \hrule height1.2pt
  \vskip 1.2pt
  \hrule height0.5pt
%  \vspace*{30\p@}%
  {\parindent \z@ \raggedright \reset@font
         \begin{flushleft}
         {\parbox[t]{0.66\textwidth}{\textsl{\textmd{#2}}}
         }\par  
    %    \leftline{\textsl{\textmd{#2}} 
    %    }\par  
         \end{flushleft}
  }
  {\parindent \z@ \raggedright \reset@font
    \ifnum \c@secnumdepth >\m@ne
         \huge \rightline{\textsf{\textsc{\@chapapp{}} \thechapter}}
         \par
     \fi
    \vskip 30\p@
    \hrule height0.5pt
    \vskip 1pt
    \hrule height1.2pt
    \vskip 30\p@
    \begin{flushright}
      \Huge \textsf{\textbf{#1}}\par
    \end{flushright}
    \vskip 5\p@
    \begin{flushright}
         \singlespace\normalsize{\parbox[t]{0.8\textwidth}{\textsl{\textmd{#3}}}
         }\par  
    \end{flushright}
    \nobreak
    \vskip 20\p@
%    \vskip 30\p@
%    \nobreak
%    %\hrule height1.2pt
%    \vskip 50\p@
  }}

\def\@fchapter[#1]#2#3#4{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{protect#1}
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@fmakechapterhead{#2}{#3}{#4}]%
                    \else
                      \@fmakechapterhead{#2}{#3}{#4}%
                      \@afterheading
                    \fi}
\def\@fmakechapterhead#1#2#3{%
  \noindent
  \hrule height1.2pt
  \vskip 1.2pt
  \hrule height0.5pt
%  \vspace*{30\p@}%
  {\parindent \z@ \raggedright \reset@font
         \begin{flushleft}
         {\parbox[t]{0.66\textwidth}{\textsl{\textmd{#2}}}
         }\par  
    %    \leftline{\textsl{\textmd{#2}} 
    %    }\par  
         \end{flushleft}
  }
  {\parindent \z@ \raggedright \reset@font
    \ifnum \c@secnumdepth >\m@ne
         \huge \rightline{\textsf{\textsc{\@chapapp{}} \thechapter}}
         \par
     \fi
    \vskip 30\p@
    \hrule height0.5pt
    \vskip 1pt
    \hrule height1.2pt
    \vskip 30\p@
    \begin{flushright}
      \Huge \textsf{\textbf{#1}\footnote{#3}}\par
    \end{flushright}
    \vskip 30\p@
    \nobreak
    %\hrule height1.2pt
    %\vskip 50\p@
  }}

\def\apdx#1{\thispagestyle{plain}
                    \ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect%\numberline{\thechapter}
                                    #1}%
                    \else
                      \addcontentsline{toc}{chapter}{protect#1}
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\makeapdx{#1}]%
                    \else
                      \makeapdx{#1}%
                      \@afterheading
                    \fi}

\def\makeapdx#1{%
  \noindent
  {\parindent \z@ \raggedright \reset@font
    \begin{flushleft}
      \Huge \textsf{\textbf{#1}}\par
    \end{flushleft}
    \vskip 15\p@
    \hrule height1.2pt
    \vskip 20\p@
    \nobreak
  }}

\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \reset@font
    \Huge \bfseries  #1\par
    \vskip 30\p@ \hrule height1.2pt
    \nobreak
    \vskip 50\p@
  }}

\newenvironment{sirabibliography}[1]
     {%\chapter*{\bibname
      %  \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\reset@font\large\bfseries}}
%               {\reset@font\sffamily\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\bfseries}}
%                                     {\reset@font\sffamily\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\normalsize\bfseries}}
%                                     {\reset@font\sffamily\normalsize\bfseries}}
\renewcommand\contentsname{\textsf{Contents}}
\renewcommand\listfigurename{\textsf{List of Figures}}
\renewcommand\listtablename{\textsf{List of Tables}}
\renewcommand\bibname{\textsf{References}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of "book.cls" Modification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{doublespace}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin of "sythesis.sty" Modification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Document Style Option `sythesis' <13 Nov 93>.}

\long\def\comment#1{}
\comment{

  Example of use:
    \documentstyle[11pt,sythesis]{book}
    \begin{document}
    \title{How to Write Theses\\
            With Two Line Titles}
    \author{John Henry Candidate}
    \principaladviser{John Parker}
    \firstreader{John Green}
    \secondreader{John BigBooty\\(Another Department)}
 
    \beforepreface
    \prefacesection{Preface}
        This thesis tells you all you need to know about...
    \prefacesection{Acknowledgements}
        I would like to thank...
    \afterpreface
 
    \chapter{Introduction}
         ...
    \chapter{Conclusions}
         ...
    \appendix
    \chapter{A Long Proof}
         ...
    \bibliographystyle{plain}
    \bibliography{mybib}
    \end{document}

Documentation:
    Dissertations".  It sets the margins and interline spacing and
    disallows page breaks at hyphens.

    The \beforepreface command creates the title page, a copyright page
    (optionally), and a signature page.  Then the user should put
    preface section(s), using the \prefacesection{section title}
    command.  The \afterpreface command then produces the tables of
    contents, tables and figures, and sets things up to start
    the main body (on arabic page 1).
    
    The following commands can control what goes in the front matter
    material:
    
        \title{thesis title}
        \author{author's name}
        \dept{author's department}
                - Computer Science if omitted
        \principaladviser{the principal advisor's name}
           (or \principaladvisor, if you prefer advisor spelled with o)
        \firstreader{the first reader's name}
        \secondreader{the second reader's name}
        \submitdate{month year in which submitted to GPO}
                - date LaTeX'd if omitted
        \copyrightyear{year degree conferred (next year if submitted in Dec.)}
                - year LaTeX'd (or next year, in December) if omitted
        \copyrighttrue or \copyrightfalse
                - produce or don't produce a copyright page (true by default)
        \figurespagetrue or \figurespagefalse
                - produce or don't produce a List of Figures page
                  (true by default)
        \tablespagetrue or \tablespagefalse
                - produce or don't produce a List of Tables page
                  (true by default)

This style uses interline spacing that is 1.3 times normal, except
in the figure and table environments where normal spacing is used.
That can be changed by doing:
    \renewcommand{\baselinestretch}{1.6}
(or whatever you want instead of 1.6)
Unfortunately, that command probably won't take effect unless it comes
before the \begin{document} or after something like \small \normalsize.

The example above shows the 12pt substyle being used.  This seems to give
acceptable looking results, but it may be omitted to get smaller print.

}

% First thing we do is make sure that report has been loaded.  A
% common error is to try to use suthesis as a documentstyle.
\@ifundefined{chapter}{\@latexerr{The `suthesis' option should be used
with the `report' document style}{You should probably read the
sythesis documentation.}}{}

% We need 1" margins except on the binding edge, where it is 1 1/2"
% Theses are single-sided, so we don't care about \evensidemargin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\oddsidemargin 0.5in
\evensidemargin 0.145in
\marginparwidth 40pt
\marginparsep 10pt
\topmargin 10pt
\headsep .5in

\textheight 7.6in
\textwidth 5.6in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A4 \textheight 8.1in \textwidth 6in  

% Disallow page breaks at hyphens (this will give some underfull vbox's,
% so an alternative is to use \brokenpenalty=100 and manually search
% for and fix such page breaks)
%\brokenpenalty=1000
\brokenpenalty=10000  % -> original

% Use 1.4 times the normal baseline-to-baseline skip
\renewcommand{\baselinestretch}{1.4}


% Redefine the macro used for floats (including figures and tables)
% so that single spacing is used.
% (Note \def\figure{\@float{figure}set single spacing} doesn't work
%  because figure has an optional argument)
% --- Removed legacy \@xfloat redefinition (incompatible with modern LaTeX) ---
% --- This was breaking ALL figure/table environments ---
% \def\@xfloat#1[#2]{\ifhmode \@bsphack\@floatpenalty -\@Mii\else
%    \@floatpenalty-\@Miii\fi\def\@captype{#1}\ifinner
%       \@parmoderr\@floatpenalty\z@
%     \else\@next\@currbox\@freelist{\@tempcnta\csname ftype@#1\endcsname
%        \multiply\@tempcnta\@xxxii\advance\@tempcnta\sixt@@n
%        \@tfor \@tempa :=#2\do
%                         {\if\@tempa h\advance\@tempcnta \@ne\fi
%                          \if\@tempa t\advance\@tempcnta \tw@\fi
%                          \if\@tempa b\advance\@tempcnta 4\relax\fi
%                          \if\@tempa p\advance\@tempcnta 8\relax\fi
%          }\global\count\@currbox\@tempcnta}\@fltovf\fi
%     \global\setbox\@currbox\vbox\bgroup 
%     \def\baselinestretch{1}\@normalsize
%     \boxmaxdepth\z@
%     \hsize\columnwidth \@parboxrestore}
% --- End of removed block ---
% Redefine the macro used for footnotes to use single spacing
\long\def\@footnotetext#1{\insert\footins{\def\baselinestretch{1}\footnotesize
    \interlinepenalty\interfootnotelinepenalty 
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
   \edef\@currentlabel{\csname p@footnote\endcsname\@thefnmark}\@makefntext
    {\rule{\z@}{\footnotesep}\ignorespaces
      #1\strut}}}

% \author, \title are defined in report; here are the rest of the
% front matter defining macros
\def\dept#1{\gdef\@dept{#1}}
\def\principaladviser#1{\gdef\@principaladviser{#1}}
\def\advis@r{Adviser} % default spelling
\def\principaladvisor#1{\gdef\@principaladviser{#1}\gdef\advis@r{Advisor}}
\def\firstreader#1{\gdef\@firstreader{#1}}
\def\secondreader#1{\gdef\@secondreader{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}} % \author, \title in report
\def\@title{}\def\@author{}\def\@dept{computer science}
\def\@principaladviser{}\def\@firstreader{}\def\@secondreader{}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\the\year}
% Stanford says that December theses should have the next year as the
% copyright year
\ifnum\month=12
    \@tempcnta=\year \advance\@tempcnta by 1
    \edef\@copyrightyear{\number\the\@tempcnta}
\else
    \def\@copyrightyear{\number\the\year}
\fi
\newif\ifcopyright \newif\iffigurespage \newif\iftablespage
\copyrighttrue \figurespagetrue \tablespagetrue
%%%%%%%%%%%%%%%%[ Title ]%%%%%%%%%%%%%%%
\def\titlep{%
  \thispagestyle{empty}
  \null\vskip.6in
  \begin{singlespace}
    \rmfamily\large
    \begin{center}
      \textsf{\textbf{\Huge\@title}}
    \end{center}
    \vskip0.5in
    \begin{center}
      Advisor : {\Large \@principaladviser} \\
      \vskip.5in
      By \\
      \bigskip
      {\Large \@author} \\
      \bigskip
      Department of Electrical Engineering \\
      Korea Advanced Institute of Science and Technology
    \end{center}
    \vskip.6in
%    \begin{center}
%      \@submitdate
%    \end{center}
    \vskip.3in
    \textsl{\textmd{%
      A thesis submitted to the faculty of the Korea Advanced Institute
      of Science and Technology    
      in partial fulfillment of the requirements
      for the degree of doctor of Philosophy in the Department of
      Electrical Engineering.}}
  \end{singlespace}
  \newpage
  }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\signaturepage{%
        \signature{\@principaladviser\\(Principal \advis@r)}
        \vfill
        \signature\@firstreader
        \vfill
        \signature\@secondreader
        \vfill
        \begin{center}
        \begin{minipage}{4in}
        Approved for the University Committee on Graduate Studies:\par
        \vspace{.5in}
        \hbox to 4in{\hfil\shortstack{\vrule width 3in height 0.4pt\\
                        Dean of Graduate Studies}}
        \end{minipage}
        \end{center}}
\def\beforepreface{
%        \pagenumbering{roman}
%        \pagestyle{plain}
%        \thispagestyle{empty}
        \titlep
%       \signaturepage
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\abstractsection{
        \null\vspace*{50pt}
        \begin{center}
          \textsf{\textbf{\LARGE Abstract}}
        \end{center}
        \null\vskip 20pt
        \addcontentsline{toc}{chapter}{Abstract}
}
\def\prefacesection#1{%
        \chapter*{#1}
        \addcontentsline{toc}{chapter}{#1}}
\def\afterpreface{\newpage
        \tableofcontents
        \newpage
        \pagestyle{plain}
}
\def\mylistoffigures{\newpage
        \iffigurespage
                \listoffigures
                \addcontentsline{toc}{chapter}{List of Figures}
                \newpage
        \fi
}
\comment{ by okman
\def\afterpreface{\newpage
        \tableofcontents
        \newpage
        \iftablespage
                \addcontentsline{toc}{chapter}{\textrm{List of Tables}}
                \listoftables
                \newpage
        \fi
        \iffigurespage
                \addcontentsline{toc}{chapter}{\textrm{List of Figures}}
                \listoffigures
                \newpage
        \fi
        \pagenumbering{arabic}
        \pagestyle{plain}
}
}
% titlepage.sty  27 Jan 85
\def\maketitle{
  \begin{titlepage}
    \let\footnotesize\small \let\footnoterule\relax \setcounter{page}{0}
    \null\vfil
    \vskip 60pt
    \begin{center}
      {\LARGE \@title \par} \vskip 3em
      {\Large \lineskip .75em \begin{tabular}[t]{c}\@author \end{tabular}\par}
      \vskip 1.5em {\large \@date \par}
    \end{center} \par
    \@thanks
    \vfil
    \null
  \end{titlepage}
  \setcounter{footnote}{0} \let\thanks\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\maketitle\relax
} 

\def\abstract{
%       \titlepage
%       \null\vfil
  \null\vskip0.5in
  \begin{center}
    {\LARGE \bfseries Abstract}
  \end{center}
}

\def\endabstract{
  \par\vfil\null\endtitlepage
}


% Redefine \thebibliography to go to a new page and put an entry in the
% table of contents
\let\@ldthebibliography\thebibliography
\renewcommand{\thebibliography}[1]{\newpage
        \addcontentsline{toc}{chapter}{References}
        \@ldthebibliography{#1}}

% Start out normal
\pagestyle{plain}

%by Dr. Kim 2005 12 12
% \setcounter{bottomnumber}{1}

\sloppy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of "sythesis.sty" Modification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
