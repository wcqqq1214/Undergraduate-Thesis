\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gxuthesis}[2026/04/04 Guangxi University Thesis Class]



% —— 基础文档类 ——
\LoadClass[UTF8, 12pt, fontset=none]{ctexart}
% fontset 必须是 none，因为所有字体是模板自行定义的



% —— 宏包加载 ——
\RequirePackage{comment}
\RequirePackage{silence}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\WarningFilter{fancyhdr}{\headheight is too small}
\RequirePackage{tikz}
\RequirePackage{ulem}
\RequirePackage{setspace}
\RequirePackage{tocloft}
\RequirePackage[nottoc]{tocbibind}
\RequirePackage{calc}
\RequirePackage{xpatch}
\RequirePackage{etoolbox}
\RequirePackage{appendix}
\RequirePackage{indentfirst}
\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{array}
\RequirePackage{color}
\RequirePackage{pdfpages}
% amsmath 必须最先显式加载，防止 cases/amsthm 触发二次加载与 unicode-math 冲突；unicode-math 检测到 amsmath 已加载，跳过重定义，消除冲突
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{cases}
\RequirePackage{braket}
\RequirePackage[warnings-off={mathtools-colon,mathtools-overbracket}]{unicode-math}
\RequirePackage{textgreek}
\RequirePackage{fontspec}
\RequirePackage{caption}
\RequirePackage{bicaption}
\RequirePackage{lipsum}
\RequirePackage{zhlipsum}
\RequirePackage[colorlinks, linkcolor=black, anchorcolor=black, citecolor=black, urlcolor=black]{hyperref}
\RequirePackage{cleveref}

\newif\ifgxugbtwentyfive
\gxugbtwentyfivefalse
\DeclareOption{gb7714-2025}{\gxugbtwentyfivefalse\gxugbtwentyfivefalse
\gxugbtwentyfivetrue}
\ProcessOptions\relax
\ifgxugbtwentyfive
    \RequirePackage[
        backend=biber,
        style=gb7714-2025,
        gbpub=true,
        gbnamefmt=lowercase,
        doi=false,
        url=false,
        gbalign=left,
        gbpunctwidth=bylan,
    ]{biblatex}
    \setcounter{gbrefcompress}{3}
    \def\gbpunctcommacite{\addcomma}
    \DeclareFieldFormat{title}{#1}
    \DeclareFieldFormat[article,patent,thesis,unpublished]{title}{#1}
    \makeatletter
    \AtBeginDocument{\patchcmd{\abx@macro@title}{\newunitpunct}{}{}{}}{}
    \makeatother
    \def\gbpunctdot{\unskip\mbox{．}\allowbreak}
\else
    \RequirePackage[
        backend=biber,
        style=gb7714-2015,
        gbpub=true,
        gbnamefmt=lowercase,
        doi=false,
        url=false,
        gbalign=left,
    ]{biblatex}
    \setcounter{gbrefcompress}{3}
\fi



% —— 字体设置 ——
% 优先系统字体，其次本地字体文件，最后回退

% 宋体
\IfFontExistsTF{SimSun}{%
    \setCJKmainfont{SimSun}[AutoFakeBold=3, AutoFakeSlant=0.25]%
    \newfontfamily\simsun{SimSun}%
    \setCJKfamilyfont{songti}{SimSun}%
}{%
    \IfFileExists{fonts/simsun.ttc}{%
        \setCJKmainfont{simsun.ttc}[AutoFakeBold=3, AutoFakeSlant=0.25, Path=fonts/]%
        \newfontfamily\simsun{simsun.ttc}[Path=fonts/]%
        \setCJKfamilyfont{songti}{simsun.ttc}[Path=fonts/]%
    }{%
        \IfFileExists{fonts/simsun.ttf}{%
            \setCJKmainfont{simsun.ttf}[AutoFakeBold=3, AutoFakeSlant=0.25, Path=fonts/]%
            \newfontfamily\simsun{simsun.ttf}[Path=fonts/]%
            \setCJKfamilyfont{songti}{simsun.ttf}[Path=fonts/]%
        }{%
            \PackageWarning{gxuthesis}{SimSun not found, using ctex default songti}%
            \newcommand{\simsun}{\rmfamily}%
        }%
    }%
}
\newcommand{\songti}{\CJKfamily{songti}}

% 黑体
\IfFontExistsTF{SimHei}{%
    \setCJKfamilyfont{heiti}{SimHei}[AutoFakeBold=3, AutoFakeSlant=0.25]%
}{%
    \IfFileExists{fonts/simhei.ttf}{%
        \setCJKfamilyfont{heiti}{simhei.ttf}[AutoFakeBold=3, AutoFakeSlant=0.25, Path=fonts/]%
    }{%
        \PackageWarning{gxuthesis}{SimHei not found, using ctex default heiti}%
    }%
}
\newcommand{\heiti}{\CJKfamily{heiti}}

% 隶书
\newif\ifgxulishufound
\gxulishufoundfalse
\IfFontExistsTF{SimLi}{%
    \setCJKfamilyfont{lishu}{SimLi}%
    \gxulishufoundtrue
}{%
    \IfFileExists{fonts/simli.ttf}{%
        \setCJKfamilyfont{lishu}{simli.ttf}[Path=fonts/]%
        \gxulishufoundtrue
    }{%
        \PackageWarning{gxuthesis}{SimLi not found, falling back to heiti bold}%
    }%
}
\ifgxulishufound
    \newcommand{\lishu}{\CJKfamily{lishu}}%
\else
    \newcommand{\lishu}{\bfseries\heiti}
\fi

% 仿宋
\IfFontExistsTF{FangSong}{%
    \setCJKfamilyfont{zhfs}{FangSong}%
    \setCJKmonofont{FangSong}%
}{%
    \IfFileExists{fonts/fangsong.ttf}{%
        \setCJKfamilyfont{zhfs}{fangsong.ttf}[Path=fonts/]%
        \setCJKmonofont{fangsong.ttf}[Path=fonts/]%
    }{%
        \PackageWarning{gxuthesis}{FangSong not found, using ctex default fangsong}%
    }%
}
\providecommand{\fangsong}{}
\renewcommand{\fangsong}{\CJKfamily{zhfs}}

% Times New Roman
\IfFontExistsTF{Times New Roman}{%
    \setmainfont{Times New Roman}%
}{%
    \IfFileExists{fonts/times.ttf}{%
        \setmainfont{times.ttf}[%
            Path           = fonts/,%
            BoldFont       = timesbd.ttf,%
            ItalicFont     = timesi.ttf,%
            BoldItalicFont = timesbi.ttf%
        ]%
    }{%
        \PackageWarning{gxuthesis}{Times New Roman not found, using Latin Modern}%
    }%
}%


% 数学字体
\setmathfont{XITS Math}[
    math-style = ISO,
    bold-style  = ISO,
    StylisticSet = 8
]

% 取消这段注释，使用真正的新罗马作为数学字体
% 效果一般，不好看，容易出问题
\begin{comment}
\setmathfont{Times New Roman}[
    range = {up, "0030-"0039, "0041-"005A, "0061-"007A},
]
\setmathfont{Times New Roman Italic}[range = {it}]
\setmathfont{Times New Roman Bold}[range = {bfup}]
\setmathfont{Times New Roman Bold Italic}[range = {bfit}]
\setmathfont{XITS Math}[range={
    "2202,  % \partial
    "1D716, % \epsilon
    "1D717, % \vartheta
    "1D71A, % \varrho
    "1D71B, % \varpi
    "1D719, % \phi
    "2207,  % \nabla
}]
\end{comment}

% 板粗体与花体
\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}
\let\mathbb\relax
\DeclareMathAlphabet{\mathbb}{U}{msb}{m}{n}



% —— 数学命令 ——
\providecommand{\square}{\mdlgwhtsquare}
\newcommand{\bm}[1]{\symbf{#1}}
\providecommand{\dd}{\mathop{}\!\mathrm{d}}
\providecommand{\dif}{\mathop{}\!\mathrm{d}}
\providecommand{\ee}{\mathrm{e}}
\providecommand{\ii}{\mathrm{i}}
\providecommand{\jj}{\mathrm{j}}
\renewcommand\mathellipsis{\cdots}

\xeCJKsetup{CJKmath=true}

\AtBeginDocument{
    \renewcommand{\bm}[1]{\symbf{#1}}
    \renewcommand{\mathbf}[1]{\symbf{#1}}
    \renewcommand{\mathrm}[1]{\symup{#1}}
    \renewcommand{\mathit}[1]{\symit{#1}}
    \renewcommand{\Re}{\operatorname{Re}}
    \renewcommand{\Im}{\operatorname{Im}}
    % 大写希腊字母改为默认直立，\varXxx 为斜体
    \let\origGamma\Gamma   \let\origDelta\Delta     \let\origTheta\Theta
    \let\origLambda\Lambda \let\origXi\Xi           \let\origPi\Pi
    \let\origSigma\Sigma   \let\origUpsilon\Upsilon \let\origPhi\Phi
    \let\origPsi\Psi       \let\origOmega\Omega
    \renewcommand{\Gamma}{\symup{\origGamma}}     \renewcommand{\varGamma}{\origGamma}
    \renewcommand{\Delta}{\symup{\origDelta}}     \renewcommand{\varDelta}{\origDelta}
    \renewcommand{\Theta}{\symup{\origTheta}}     \renewcommand{\varTheta}{\origTheta}
    \renewcommand{\Lambda}{\symup{\origLambda}}   \renewcommand{\varLambda}{\origLambda}
    \renewcommand{\Xi}{\symup{\origXi}}           \renewcommand{\varXi}{\origXi}
    \renewcommand{\Pi}{\symup{\origPi}}           \renewcommand{\varPi}{\origPi}
    \renewcommand{\Sigma}{\symup{\origSigma}}     \renewcommand{\varSigma}{\origSigma}
    \renewcommand{\Upsilon}{\symup{\origUpsilon}} \renewcommand{\varUpsilon}{\origUpsilon}
    \renewcommand{\Phi}{\symup{\origPhi}}         \renewcommand{\varPhi}{\origPhi}
    \renewcommand{\Psi}{\symup{\origPsi}}         \renewcommand{\varPsi}{\origPsi}
    \renewcommand{\Omega}{\symup{\origOmega}}     \renewcommand{\varOmega}{\origOmega}
}

% 上下标缩放微调以匹配 Word，上标调了，下标我没太细调高度
\let\oldsuperscript\textsuperscript
\let\oldsubscript\textsubscript
\renewcommand{\textsuperscript}[1]{%
    \oldsuperscript{\raisebox{-0.15ex}{\scalebox{0.925}{#1}}}}
\renewcommand{\textsubscript}[1]{%
    \oldsubscript{\raisebox{0.55ex}{\scalebox{0.925}{#1}}}}



% —— 页面版式 ——
\geometry{
    papersize={21cm, 29.7cm},
    left=2.8cm, right=2.5cm, top=2.8cm, bottom=2.695cm,
    includefoot,
    headsep=7.75pt,
    footskip=.495cm,
}

% 1.3475 对应 Word 1.25倍行距
% 1.731 则对应 word 25 磅
% 均指在「文档网格-只指定行网格-每页 44行」下的表现
\newcommand{\gxustretchnormal}{\setstretch{1.3475}} % 1.25 倍行距
\newcommand{\gxustretchwide}{\setstretch{1.731}} % 25 磅行距
\gxustretchnormal
\raggedbottom
\setlength{\biblabelsep}{0.5em}

% 节标题格式
\newcommand{\gxuafternamewide}{%
    \ctexset{
        section/aftername=\hspace{1em},
        subsection/aftername=\hspace{1em},
        subsubsection/aftername=\hspace{1em},
        paragraph/aftername=\hspace{1em},
    }%
}
\newcommand{\gxuafternamenarrrow}{%
    \ctexset{
        section/aftername=\hspace{.5em},
        subsection/aftername=\hspace{.5em},
        subsubsection/aftername=\hspace{.5em},
        paragraph/aftername=\hspace{.5em},
    }%
}
\ctexset{
    contentsname  = \heiti\zihao{3}\textbf{目\hspace{.5em}录},
    section={
        name={第,章}, number=\chinese{section},
        format={\heiti\bfseries\centering\zihao{3}\linespread{1.6331}\selectfont}, % > 1.633 < 1.6331，似乎不能更精细了，精度是三行文字之内无误差，足矣
        aftername=\hspace{1em},
        beforeskip=0pt, afterskip=17.75pt
    },
    subsection={
        format=\heiti\bfseries\zihao{4},
        aftername=\hspace{1em},
        beforeskip=6.75pt, afterskip=9pt
    },
    subsubsection={
        format=\heiti\bfseries\zihao{-4},
        aftername=\hspace{1em},
        beforeskip=0pt, afterskip=0pt
    },
    paragraph={
        runin=false,
        format=\normalfont\normalsize,
        aftername=\hspace{1em},
        beforeskip=0pt, afterskip=0pt, indent=0pt,
        numbering=true,
        number=\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.\arabic{paragraph},
        name={,},
    },
    appendix/name=附录
}

% section 前额外 -0.75pt 补偿
\let\oldsection\section
\renewcommand{\section}{\vspace*{-0.75pt}\oldsection}

% subsection 前若紧跟 section，额外 -0.75pt
\makeatletter
\newif\if@gxu@insection
\@gxu@insectionfalse
\let\gxu@orig@startsection\@startsection
\def\@startsection#1#2#3#4#5#6{%
    \ifnum#2=2\relax\if@gxu@insection\vskip-0.75pt\fi\fi
    \ifnum#2=1\relax\global\@gxu@insectiontrue
    \else\global\@gxu@insectionfalse\fi
    \gxu@orig@startsection{#1}{#2}{#3}{#4}{#5}{#6}}
\makeatother



% —— 目录格式 ——
\setcounter{secnumdepth}{4} % 目录深度到 x.x.x
\renewcommand{\cfttoctitlefont}{\hfill}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecfont}{\normalfont}
\renewcommand{\cftsecpagefont}{\normalfont}
\setlength{\cftbeforesecskip}{0pt}
\renewcommand{\cftdotsep}{.5}
\setlength{\cftsecindent}{0em}      \setlength{\cftsecnumwidth}{0em}
\setlength{\cftsubsecindent}{2em}   \setlength{\cftsubsecnumwidth}{0em}
\setlength{\cftsubsubsecindent}{4em}\setlength{\cftsubsubsecnumwidth}{0em}

\makeatletter
\newcommand{\gxulistoffigures}{%
    \section*{插图清单}%
    \addcontentsline{toc}{section}{插图清单}%
    \@starttoc{lof}%
}

\newcommand{\gxulistoftables}{%
    \section*{附表清单}%
    \addcontentsline{toc}{section}{附表清单}%
    \@starttoc{lot}%
}
\makeatother

% 图清单
\setlength{\cftfignumwidth}{.5em}
% 表清单
\setlength{\cfttabnumwidth}{.5em}



% —— 图表题注格式 ——
\graphicspath{{图/}{./}}

\captionsetup[figure][bi]{labelfont=bf, textfont=bf}
\captionsetup[figure][bi-second]{name=Figure, labelfont=bf, textfont=bf}
\captionsetup[figure]{skip=3.5bp, belowskip=-6.75bp}

\captionsetup[table][bi]{labelfont=bf, textfont=bf}
\captionsetup[table][bi-second]{name=Table, labelfont=bf, textfont=bf}
\captionsetup[table]{skip=15bp, belowskip=-2.25bp}

\DeclareCaptionFont{wuhaozhutu}{\fontsize{10.5bp}{17.625bp}\selectfont}
\DeclareCaptionFont{wuhaozhubiao}{\fontsize{10.5bp}{15.25bp}\selectfont}

\captionsetup[figure]{font=wuhaozhutu, labelsep=space, justification=centering}
\captionsetup[table]{font=wuhaozhubiao, labelsep=space, justification=centering}

\captionsetup[figure][bi-second]{list=no}
\captionsetup[table][bi-second]{list=no}

\newcommand{\fignote}[1]{%
    \vspace{23.25pt}
    \begin{minipage}{\linewidth}%
        \fontsize{10.5bp}{18.85bp}\selectfont\hspace{2em}注：#1%
    \end{minipage}
    \vspace{-21.5pt}
}

% —— 页眉页脚 ——
% 根据标题行数自动调整版面，支持一行/两行，再多不行了
\newlength{\gxuheadertitleheight} % 页眉高度测量
\newcommand{\setheadertitle}[1]{%
    \ifgxublind
        \newgeometry{
            left=2.8cm, right=2.5cm, top=2.8cm, bottom=2.695cm,
            includefoot, headsep=7.75pt, footskip=.495cm}%
        \renewcommand{\headrule}{%
            \vspace{4pt}\hrule width\headwidth height\headrulewidth\vspace{-4pt}}%
        \fancyhead[R]{}%
        \renewcommand{\headrulewidth}{0pt}% 盲审不要横线
    \else
        \begingroup
        \def\titlebreak{}%
        \settototalheight{\gxuheadertitleheight}{%
            \parbox{\dimexpr\textwidth-4cm}{%
                \lishu\fontsize{12bp}{19.5pt}\selectfont\ziju{.05}#1}}%
        \ifdim\gxuheadertitleheight>20pt
            \endgroup
            \newgeometry{
                left=2.8cm, right=2.5cm, top=3.46cm, bottom=2.695cm,
                includefoot, headsep=27.75pt, footskip=.495cm}%
            \renewcommand{\headrule}{%
                \vspace{7.75pt}\hrule width\headwidth height\headrulewidth\vspace{-7.75pt}}%
        \else
            \endgroup
            \newgeometry{
                left=2.8cm, right=2.5cm, top=2.8cm, bottom=2.695cm,
                includefoot, headsep=7.75pt, footskip=.495cm}%
            \renewcommand{\headrule}{%
                \vspace{4pt}\hrule width\headwidth height\headrulewidth\vspace{-4pt}}%
        \fi
        \fancyhead[R]{\vtop{%
            \def\titlebreak{}%
            \lishu\fontsize{12bp}{19.5pt}\selectfont\ziju{.05}#1}}%
    \fi
}



% —— 摘要 ——
\newcommand{\cnabstractname}{\heiti\fontsize{16bp}{20pt}\textbf{摘\hspace{.5em}要}}
\newcommand{\enabstractname}{\fontsize{16bp}{20pt}\textbf{ABSTRACT}}

\newenvironment{cnabstract}{%
    \par\small
    \noindent\mbox{}\hfill{\bfseries\cnabstractname}\hfill\mbox{}%
    \par\vskip 2.5ex
}{\par\vskip 2.5ex}

\newenvironment{enabstract}{%
    \par\small
    \tolerance=2000\emergencystretch=3em
    \noindent\mbox{}\hfill{\bfseries\enabstractname}\hfill\mbox{}%
    \par\vskip 2.5ex
}{\par\vskip 2.5ex}

% 摘要节标题（无编号，格式与正文 section 一致）
\newlength{\sectiontitleheight}
\newcommand{\cnabstractsection}[1]{%
    \vspace*{-9.5pt}\vspace*{-11.5pt}%
    {\ctexset{section/format={\heiti\bfseries\centering\zihao{3}\linespread{2.165}\selectfont}}%
    \def\titlebreak{\\}%
    \section*{#1}}%
}

% 关键词由摘要子文件（或主文件）用 \newcommand 定义：
%   \newcommand{\cnkeywords}{词1, 词2, 词3}
%   \newcommand{\enkeywords}{word1, word2, word3}
% cls 不预定义，避免与子文件的 \newcommand 冲突。

% 关键词打印命令
\RequirePackage{listofitems}
\newcommand{\printcnkeywords}[1]{%
    \noindent{\heiti\textbf{关键词：}}\songti
    \setsepchar{,}%
    \readlist\kwlist{#1}%
    \foreachitem\kw\in\kwlist{%
        \ifnum\kwcnt>1\hspace{1em}\fi\kw}}

\newcommand{\printenkeywords}[1]{%
    \noindent{\textbf{KEYWORDS: }}%
    \setsepchar{,}%
    \readlist\kwlist{#1}%
    \foreachitem\kw\in\kwlist{%
        \ifnum\kwcnt>1;\space\fi\kw}}



% —— 前置页 —— 封面、摘要、目录
\newcommand{\makefrontmatter}[4]{%
    % —— 封面 ——
    \setcounter{page}{1}
    \pagenumbering{roman}
    \makecover
    \ifgxustatement\makestatement\fi

    % —— 页脚样式（摘要/目录区段）——
    \fancypagestyle{plain}{%
        \fancyhf{}
        \fancyfoot[C]{\zihao{-5}\raisebox{-12.625pt}{\thepage}} % 这个管主目录的页码
        \renewcommand{\headrulewidth}{0mm}}
    \setcounter{page}{1}
    \pagenumbering{Roman}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyfoot[C]{\zihao{-5}\raisebox{-12.625pt}{\thepage}} % 这个管其他罗马数字
    \renewcommand{\headrulewidth}{0mm}

    % —— 中文摘要 ——
    \cnabstractsection{#1}
    \vspace{10.75pt}
    \begin{cnabstract}\addcontentsline{toc}{section}{摘要}
        \fontsize{14bp}{16.85bp}\selectfont
        \vspace{4.75pt}
        \input{#3}\par
        \vspace*{6.5pt}
        \phantom{字}\\
        \printcnkeywords{\cnkeywords}
    \end{cnabstract}
    \pagebreak

    % —— 英文摘要 ——
    \vspace*{-18.1pt}
    {\ctexset{section/format={\heiti\bfseries\centering\zihao{3}\linespread{2.0165}\selectfont}}
    \def\titlebreak{\\}%
    \section*{#2}}
    \vspace*{2.75pt}
    \begin{enabstract}%
        \addtocontents{toc}{\vspace{-1pt}}%
        \addcontentsline{toc}{section}{ABSTRACT}%
        \addtocontents{toc}{\vspace{-1.25pt}}
        \fontsize{14bp}{14.9bp}\selectfont
        \vspace*{4.8pt}
        \input{#4}
        \vskip 2em\vskip 12.5pt
        \printenkeywords{\enkeywords}
    \end{enabstract}
    \pagebreak

    % —— 目录 ——
    \setlength{\cftaftertoctitleskip}{17.75pt}
    \vspace*{-18.5pt}
    {
    \begin{spacing}{1.3475}
        \tableofcontents
    \end{spacing}}
    \restoregeometry
}

\newcommand{\gxulistofabbr}{%
    \section*{缩略语清单}%
    \addcontentsline{toc}{section}{缩略语清单}%
}

\newcommand{\gxulistofsymbols}{%
    \section*{符号清单}%
    \addcontentsline{toc}{section}{符号清单}%
}

\newcommand{\gxulistofterms}{%
    \section*{术语表}%
    \addcontentsline{toc}{section}{术语表}%
}



% —— 正文部分 —— 改页眉
\newcommand{\startmainmatter}{%
    \newpage
    \setcounter{page}{1}
    \pagenumbering{arabic}
    \pagestyle{fancy}
    \fancyhf{}
    \ifgxublind
        \fancyhead[L]{}%
    \else
        \fancyhead[L]{\vtop{\lishu\fontsize{12bp}{19.5pt}\selectfont 广西大学\degree 学位论文}}%
    \fi
    \setheadertitle{\cntitle}
    \fancyfoot[C]{\zihao{-5}\thepage}
    \ifgxublind
        \renewcommand{\headrulewidth}{0pt}%
    \else
        \renewcommand{\headrulewidth}{0.5pt}%
    \fi
    \numberwithin{equation}{section}
    \numberwithin{table}{section}
    \numberwithin{figure}{section}
    \renewcommand{\thetable}{\thesection-\arabic{table}}
    \renewcommand{\thefigure}{\thesection-\arabic{figure}}
    \renewcommand{\theequation}{\thesection-\arabic{equation}}
}

% —— 列表 ——
\RequirePackage{enumitem}

% 无序列表
\setlist[itemize]{%
  itemsep=0pt, % 每一项之间的垂直间距
  topsep=0pt, % 列表上下方与正文的垂直间距
  parsep=0pt, % 段落间距
  partopsep=0pt, % 段落顶部附加间距
  left=2em % 缩进距离（控制整体左移或右移）
}

% 有序列表
\RequirePackage{multicol}

\setlist[enumerate]{%
  itemsep=0pt,
  topsep=0pt,
  parsep=0pt,
  partopsep=0pt,
  left=2em
}

\SetEnumitemKey{shuzi}{
    label=\arabic*.,
    listparindent=2em,
    labelsep=0.25em,
    itemindent=3em
}

\SetEnumitemKey{kuohao}{
    label=（\arabic*）,
    listparindent=2em,
    labelsep=0pt,
    itemindent=4em
}

\RequirePackage{alphalph}

% 处理enumitem星号机制
\makeatletter
\newcommand{\AlphAlphEnum}[1]{\AlphAlph{\value{#1}}}
\newcommand{\alphalphEnum}[1]{\alphalph{\value{#1}}}

% 注册为enumitem的计数器
\AddEnumerateCounter{\AlphAlphEnum}{\AlphAlph}{AA}
\AddEnumerateCounter{\alphalphEnum}{\alphalph}{aa}
\makeatother

% 定义键
\SetEnumitemKey{daxie}{
    label=\AlphAlphEnum*,
    leftmargin=0pt,
    labelindent=0pt,
    labelsep=0.5em,
    itemindent=3.25em
}
\SetEnumitemKey{xiaoxie}{
    label=\alphalphEnum*,
    leftmargin=0pt,
    labelindent=0pt,
    labelsep=0.5em,
    itemindent=3.25em
}

% 参考文献
\setlength{\bibitemsep}{0pt}




% —— 附录编号重置 ——
\newcommand{\resetappendixcounters}{%
    \setcounter{table}{0}
    \setcounter{figure}{0}
    \setcounter{equation}{0}
    \renewcommand{\thetable}{\thesection-\arabic{table}}
    \renewcommand{\theequation}{\thesection-\arabic{equation}}
    \renewcommand{\thefigure}{\thesection-\arabic{figure}}
    \ctexset{section={name={附录}, number=\thesection}}%
}

\RequirePackage{imakeidx}
\makeindex[title=, intoc=false, options=-s gxuindex.ist]
\makeatletter
\newcommand{\gxuprintindex}{%
    \section*{索引}%
    \addcontentsline{toc}{section}{索引}%
    \begingroup
    \renewcommand{\clearpage}{}%
    \renewcommand{\newpage}{}%
    \printindex
    \endgroup%
}
\makeatother