%%
%%
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{snuee}
              [2010/10/21 v1.1 Standard SNU EE document class]
% 2010/10 Added "korean" option - jdj

% \documentclass에 대한 옵션 필드 설정
% Usage: \documentclass[degree]{class_file_name}
% degree는 두 가지 선택가능 master, doctor
\newif\if@Doctor
\DeclareOption{doctor} {\@Doctortrue}
\DeclareOption{master} {\@Doctorfalse}

% korean... 설정 된 경우 국문 논문임.
\newif\if@Korean
\DeclareOption{korean} {\@Koreantrue}
\DeclareOption{english} {\@Koreanfalse}


\DeclareOption{leqno}  {\PassOptionsToClass{leqno}  {report}}
\DeclareOption{fleqn}  {\PassOptionsToClass{fleqn}  {report}}
\DeclareOption{openbib}{\PassOptionsToClass{openbib}{report}}

\ExecuteOptions{doctor}
\ProcessOptions

% base class: report class 사용
% report class에 대한 설명은 latex 관련 book에서 참고가능
%\LoadClass[11pt,b5paper,onecolumn,oneside]{report}
\LoadClass[11pt,a4paper,onecolumn,oneside]{report}
\usepackage{epsfig}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{ifthen}
\usepackage{times}
\usepackage{chngpage}
\DeclareMathOperator*{\argmax}{argmax}
%
\usepackage{multirow}
\usepackage{multicol}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{hhline}
\usepackage{diagbox}
\usepackage{color}
\usepackage[usestackEOL]{stackengine}
\usepackage[skip=0pt]{caption}
\if@Korean
	\usepackage{tocloft}
	\usepackage[hangul]{kotex}
\else
	\usepackage{kotex}
\fi


\setlength{\paperwidth}  {210mm}
\setlength{\paperheight} {297mm}
%% 편집용지 설정
\usepackage[paperheight=297mm,paperwidth=210mm,
			left=40mm,right=40mm,top=57mm,bottom=57mm]{geometry}
%% 줄간격 설정
%\renewcommand{\baselinestretch}{1.3}
\linespread{1.6}
%%들여쓰기
\usepackage{indentfirst}
\setlength\parindent{2.0em}

%% 편집용지 설정(a5)
%\setlength{\paperwidth}  {182mm}
%\setlength{\paperheight} {257mm}

% 페이지여백 설정
%\addtolength{\hoffset}    {-1in}
%\addtolength{\voffset}  {-1in}
%\setlength{\topmargin}      {30mm}
%\setlength{\headheight} {0mm}
%\setlength{\headsep}    {0mm}

%\setlength{\marginparwidth} {0mm}
%\setlength{\marginparsep}   {0mm}

%\setlength{\oddsidemargin}  {25mm}
%\setlength{\textwidth}      {132mm}
%\setlength{\textheight} {197mm}
\setlength{\footskip}       {15mm}
%\if@Korean
%	\setlength{\textheight} {187mm}
%	\setlength{\footskip}   {10mm}
%\fi

% \title 명령어 재설정
% 한글, 영어 두 종류의 타이틀을 모두 넣는다.
% Usage: \title[korean]{Korean_Title}
% Usage: \title[english]{English_Title}

% renew related strings if Korean.
\if@Korean
	% very dirty and inconsistant... kscntformat sometimes does not work,
	% so I overrided them with renewcommand.
	\kscntformat{chapter}{제~}{~장}
	\kscntformat{section}{\arabic{chapter}.}{}
	\kscntformat{subsection}{\arabic{chapter}.}{}{.}{}
	\renewcommand\thesubsection{\arabic{chapter}.\arabic{section}.\arabic{subsection}}

	\ksnamedef{abstractname}{초록}
	\ksnamedef{contentsname}{목차}
	\ksnamedef{listtablename}{표 목차}
	\ksnamedef{listfigurename}{그림 목차}

	% for table of contents
	% need tocloft package
	\renewcommand\cftfigpresnum{그림 }
	\addtolength\cftfignumwidth{2.3em}
	\renewcommand\cfttabpresnum{표 }
	\addtolength\cfttabnumwidth{2.3em}
\fi

\renewcommand{\title}[2][korean]{%
    \ifthenelse{\equal{#1}{korean}}{\newcommand{\@KorTitle}{#2}}{}%
    \ifthenelse{\equal{#1}{english}}{\newcommand{\@EngTitle}{#2}}{}%
}
\@onlypreamble{\title}

% 저자 정보
\renewcommand{\author}[2][korean]{%
    \ifthenelse{\equal{#1}{korean}}{\newcommand{\@KorAuthor}{#2}}{}%
    \ifthenelse{\equal{#1}{english}}{\newcommand{\@EngAuthor}{#2}}{}%
    \ifthenelse{\equal{#1}{nospace}}{\newcommand{\@AuthorNS}{#2}}{}%
    }
\@onlypreamble{\author}

% 지도교수 정보
\newcommand{\adviser}[1]{\newcommand{\@adviser}{#1}}
\@onlypreamble{\adviser}

% 논문 제출일
\newcommand{\submissiondeadline}[1]{%
    \newcommand{\@SubmissionYear}{#1}
}
\@onlypreamble{\submissiondeadline}

% 논문 심사일
\newcommand{\examinationdate}[1]{%
    \newcommand{\@ExamYear}{#1}
}
\@onlypreamble{\examinationdate}

% 졸업년월
% 두가지 style에 대해서 모두 작성한다.
% Usage: \gradyear[style1]{FEBRYARY 2008}
% Usage: \gradyear[style2]{2008년 2월}
% 제시한 style1, style2의 형식에 꼭 맞춰야 함.
%
\newcommand{\gradyear}[2][english]{%
    \ifthenelse{\equal{#1}{english}}{\newcommand{\@GradYearEng}{#2}}{}%
    \ifthenelse{\equal{#1}{korean} }{\newcommand{\@GradYearKor}{#2}}{}%
}
\@onlypreamble{\gradyear}

% 외표지 작성
\newcommand{\makecover}{%
            \thispagestyle{empty}
            %\changetext{}{10mm}{-5mm}{}{}
%           \noindent
            \begin{center}
                \fontsize {15pt}{15pt}
                 \selectfont
				\if@Korean
					\if@Doctor
						공학박사학위논문 \\
					\else
						공학석사학위논문 \\
					\fi
				\else
					\if@Doctor
						Dissertation for the Degree of Doctor of Engineering \\
					\else
						Thesis for the Degree of Master of Engineering\\
					\fi
				\fi
                \vspace*{2cm}
                %\fontsize {22pt}{26pt}
               \fontsize {19pt}{19pt}
                \selectfont
				\if@Korean
					\@KorTitle\ \\
				\else
					\@EngTitle\ \\
				\fi
                \vspace*{1cm}
                \fontsize {16pt}{16pt}
                \selectfont
				\if@Korean
					\@EngTitle\ \\
				\else
					\@KorTitle\ \\
				\fi

				\if@Korean\else
					\vspace*{2cm}
					\fontsize {17pt}{17pt}
					 \selectfont
					by\\
					\@EngAuthor\\
					%\vspace*{2cm}
					%\Large
					%Department of IT Engineering\\
					%\Large
					%Sookmyung Women's University\\
				\fi
            \vfill
            \end{center}
                \begin{minipage}[t][30mm][b]{\textwidth}
                    \begin{center}
                    %\Large
					%\if@Korean
					%	\@GradYearKor \\
					%\else
					%	\@GradYearEng \\
					%\fi
                    %\addvspace{4cm}
					\if@Korean
						{
							\fontsize {16pt}{24pt}
							\selectfont
							숙명여자대학교 대학원\\
						}
						\vspace{2mm}
						{
							\fontsize {14pt}{16pt}
							\selectfont
							IT 공학과 \\
						}
						{
							\fontsize {16pt}{24pt}
							\selectfont
							\@KorAuthor \\
						}
					\else
					    \fontsize {17pt}{17pt}
						\selectfont
						Department of IT Engineering\\
						The Graduate School\\
						Sookmyung Women's University \\
					\fi
                    \end{center}
                \end{minipage}
            \newpage
            }

%내표지 작성
% Same as "outer" cover.
\newcommand{\makeinnercover}{%
	\makecover
}

%인준지 작성
\newcommand{\makeapproval}{%
            \thispagestyle{empty}
            %\changetext{10mm}{-10mm}{5mm}{}{}
            \noindent
            \begin{center}
                \fontsize {18.5pt}{18.5pt}
                \selectfont
				\if@Korean
					\@KorTitle \\
				\else
					\@EngTitle \\
				\fi
                \vspace{1cm}
                \fontsize {16.5pt}{16.5pt}
                \selectfont
				\if@Korean
					\@EngTitle \\
				\else
					\@KorTitle \\
				\fi
                \vspace{2.5cm}
                \fontsize{17pt}{17pt}
                \selectfont
                by\\
                \@EngAuthor \\
                \vspace{10mm}
                \fontsize {15pt}{15pt}
                \selectfont
                \if@Doctor
                    이 논문을 공학박사 학위논문으로 제출함\\
                \else
                    A Thesis submitted to the Department of IT Engineering \\
                    and the Graduate School of Sookmyung Women’s University \\
                    in partial fulfillment of the requirements for the degree of
                    Master of Engineering\\
                \fi
                \vspace{10mm}
                %\vspace{20mm}
                %\fontsize {16pt}{16pt}
                %\selectfont
                %The Graduate School\\
				%Sookmyung Women's University \\
                %\vspace{5mm}
                %\Large
                %Department of IT Engineering\\
                %\vspace {5mm}
                In charge of major work: \@adviser\\
                \vspace{10mm}
                \fontsize{17pt}{17pt}
                \selectfont
                \@SubmissionYear
                %\@GradYearEng\\
                %\if@Doctor
                %\@AuthorNS의 공학박사 학위 논문을 인준함\\
                %\else
                %\@AuthorNS의 공학석사 학위 논문을 인준함\\
                %\fi
                %\vspace{5mm}
                %\Large
                %\@GradYearKor\\
                 \vfill
            \end{center}
            \newpage
            %\begin{minipage}[t][20mm][b]{\textwidth}
            %\begin{minipage}[t][20mm][b]{\textwidth}
            \thispagestyle{empty}
                \begin{center}
                \if@Doctor
                    \Large
                    위~~원~~장:~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
                    부위원장:~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
                    위~~~~~~~~원:~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
                    위~~~~~~~~원:~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
                    위~~~~~~~~원:~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
                \else
                    %\@AuthorNS의 공학석사 학위 논문을 인준함\\
                    \fontsize {16pt}{16pt}
                    \selectfont
                \if@Korean
                   \@KorTitle \\
                \else
                   \@EngTitle \\
                \fi
                    
                    \vspace{5mm}
                    \fontsize {15pt}{15pt}
                    \selectfont
                \if@Korean
                   \@EngTitle \\
                \else
                   \@KorTitle \\
                \fi
                
                \vspace{10mm}
                \fontsize {15pt}{15pt}
                \selectfont
                    This certifies that the degree of Master of Engineering of\\
                    Yu-Jin Cho is approved by\\
                    
                    \vspace{10mm}
                    
                    \fontsize {13pt}{13pt}
                    \selectfont ~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(Signature)}
                    \vspace{-4.5mm}
                    \\\begin{flushleft}\hspace{30mm}Chair of Committee\end{flushleft}\\
                    ~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(Signature)}
                    \vspace{-4.5mm}
                    \\\begin{flushleft}\hspace{30mm}Committee Member\end{flushleft}\\
                    ~~\underline{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(Signature)}
                    \vspace{-4.5mm}
                    \\\begin{flushleft}\hspace{30mm}Committee Member\end{flushleft}\\
                    
                \vspace{10mm}
                \fontsize {17pt}{17pt}
                \selectfont
                \textbf{The Graduate School}\\
            Sookmyung Women's University \\
                \vspace{10mm}
                \@SubmissionYear \\ % 원래 GradYearEng이었는데 숙명여대_학위논문작성법.pdf에 2월졸업예정이면 December, 8월졸업예정이면 June으로 표기하도록 되어있음
                
                \fi
                \end{center}
            %\end{minipage}
            \newpage
            %\changetext{-15mm}{}{}{}{}
            }