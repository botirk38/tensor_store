\ProvidesClass{final_report}

\LoadClass[11pt, titlepage]{report}

\usepackage[
  paperwidth=21cm,
  paperheight=29.7cm,
  includehead,
  headheight=1.5cm,
  pdftex,
  hmargin={3cm,2.5cm},
  vmargin={0cm,2cm},
]{geometry}

\usepackage{fancyhdr}

\setlength{\headheight}{1.5cm}
\setlength{\parindent}{0cm}
\renewcommand{\baselinestretch}{1.2}
\parskip=1em

\renewcommand{\contentsname}{Table of Contents}

% \usepackage{fancyhdr}
% \pagestyle{fancy}
% \fancyhf{}
% \fancyhead[LO]{\small\textsf{\projecttitle}}
% \fancyhead[RO]{\small\textsf{\studentname}}
% \renewcommand{\headrulewidth}{0.1pt}
% \fancyfoot[CO]{\small\textsf{Page \thepage~of~\pageref{endpage}}}

\makeatletter

\def\ps@plain{%
  \let\@oddhead\@empty
  \def\@oddfoot{\normalfont\hfil\small\textsf{Page \thepage~of~\pageref{endpage}}\hfil}%
  \def\@evenfoot{\normalfont\hfil\small\textsf{Page \thepage~of~\pageref{endpage}}\hfil}%
}

\renewcommand{\normalsize}{\fontsize{11pt}{11pt}\selectfont}

\renewcommand{\title}[1]{\textsf{\Huge{#1}}}
\renewcommand{\author}[1]{\textsf{\Huge{#1}}}
\newcommand{\subtitle}[1]{\textsf{\textbf{\Large{#1}}}}
\newcommand{\abstractheading}[1]{\textsf{\textbf{\LARGE{#1}}}}
\newcommand{\code}[1]{\texttt{\footnotesize{#1}}}

\renewcommand{\section}{\@startsection
  {section}% name
  {1}% level
  {0mm}% indent
  {6mm}% beforeskip
  {4.2mm}% afterskip
  {\LARGE\bfseries\sffamily}% style
}

\renewcommand{\subsection}{\@startsection
  {subsection}% name
  {2}% level
  {0mm}% indent
  {4mm}% beforeskip
  {1.1mm}% afterskip
  {\Large\bfseries\sffamily}% style
}

\renewcommand{\subsubsection}{\@startsection
  {subsubsection}% name
  {3}% level
  {0mm}% indent
  {4.2mm}% beforeskip
  {1.1mm}% afterskip
  {\normalsize\bfseries\sffamily}% style
}

% \renewcommand\chapter{
%   %\thispagestyle{empty}
%   \pagestyle{fancy}
%   \doublespacing
%   \global\@topnum\z@
%   \@afterindentfalse
%   \secdef\@chapter\@schapter
% }

\renewcommand\chapter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{plain}%
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter
}

%% Chapter headings should be at the top of the page.
\def\@makechapterhead#1{%
  {%
    \parindent \z@ \raggedright \normalfont
    %\centering
    \ifnum \c@secnumdepth >\m@ne
      \huge\textsf{\@chapapp\space \thechapter:}%
      %\par\nobreak
      %\vskip 20\p@
    \fi
    \interlinepenalty\@M
    \huge \bfseries \textsf{#1}\par\nobreak
    %\rule{5cm}{0.5pt}
    \vskip 20\p@
  }%
}

\def\@makeschapterhead#1{%
  %\vspace*{50\p@}%
  {%
    \parindent \z@ \raggedright
    %\centering
    \normalfont
    \interlinepenalty\@M
    \huge \bfseries \textsf{#1}\par\nobreak
    %\rule{5cm}{0.5pt}
    \vskip 20\p@
  }%
}

\renewenvironment{abstract}{%
  \chapter*{\abstractname}%
  \addcontentsline{toc}{chapter}{\abstractname}%
}{}

\renewcommand{\l@chapter}{\bfseries\@dottedtocline{1}{0em}{2.3em}}
\renewcommand{\l@section}{\normalfont\@dottedtocline{2}{2em}{2.3em}}
\renewcommand{\l@subsection}{\normalfont\@dottedtocline{3}{2em}{2.3em}}
\renewcommand{\l@subsubsection}{\normalfont\@dottedtocline{4}{2em}{2.3em}}

\def\maketitle{%
  \begin{titlepage}
    %\thispagestyle{empty}
    \pagenumbering{gobble}
    \thispagestyle{myheadings}
    \markright{ \hfill \textnormal{\footnotesize{\studentname, \reportyear}} \hfill }
    \let\footnotesize\small
    \let\footnoterule\relax
    \setcounter{page}{0}
    \null
    \vfil
    \begin{center}
      \title{Final Year Project Report}\\[0.25cm]
      \Large \textbf{\fullOrHalfUnit~-~\finalOrInterim}\\[0.5cm]\rule{4cm}{1pt}\\[0.7cm]
      \title{\textbf{\projecttitle}}\\[1cm]
      \author{\LARGE \studentname}\\[0.5cm]\rule{4cm}{1pt}\\[0.5cm]
      \textsf{\Large A report submitted in part fulfilment of the degree of\\[0.5cm]
      \textbf{\degree}}\\[0.5cm]
      \textsf{\Large \textbf{Supervisor:} \supervisorname}\\[2cm]
      \includegraphics[height=5cm]{logo}\\[2cm]
      \textsf{\Large Department of Computer Science\\
      Royal Holloway, University of London\\\vfill
      \normalsize \today}
    \end{center}
    \vfil
    \null
  \end{titlepage}
  \pagenumbering{arabic}
}

\makeatother

\pagestyle{fancyplain}
\lhead{\footnotesize \projecttitle}
\rhead{\footnotesize \studentname}
\renewcommand{\headrulewidth}{0pt}

\setcounter{tocdepth}{1}

% \endinput  % optional, but nice to mark end of class file
