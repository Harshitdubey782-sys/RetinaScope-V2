"use client";

import { useEffect, useRef, useState, type CSSProperties, type ReactNode, type MouseEvent } from "react";

const sections = [
  {
    id: "approach",
    number: "01",
    eyebrow: "THE APPROACH",
    tag: "EXPLAINABLE SCREENING",
    title: ["Clarity before", "certainty."],
    body: "Retinal screening should make complex signals easier to understand. RetinaScope brings image analysis, explainability and structured insights into one considered workflow.",
  },
  {
    id: "technology",
    number: "02",
    eyebrow: "THE TECHNOLOGY",
    tag: "EFFICIENTNET-B0 · GRAD-CAM",
    title: ["Look closer.", "Understand more."],
    body: "An EfficientNet-B0 model analyses fundus imagery across five diabetic retinopathy grades. Grad-CAM visualisations help make the model’s attention easier to inspect.",
  },
  {
    id: "impact",
    number: "03",
    eyebrow: "THE IMPACT",
    tag: "SCREENING INTELLIGENCE",
    title: ["Earlier insight.", "Better direction."],
    body: "From individual screening results to district-level planning, RetinaScope is designed to connect screening signals with practical next steps.",
  },
];

function Reveal({ children, className = "", delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.13, rootMargin: "0px 0px -7% 0px" },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`reveal ${visible ? "reveal-visible" : ""} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

function MagneticButton({
  children,
  className = "",
  onClick,
}: {
  children: ReactNode;
  className?: string;
  onClick: () => void;
}) {
  const handleMove = (event: MouseEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left - rect.width / 2;
    const y = event.clientY - rect.top - rect.height / 2;
    event.currentTarget.style.setProperty("--mag-x", `${x * 0.12}px`);
    event.currentTarget.style.setProperty("--mag-y", `${y * 0.16}px`);
  };

  const reset = (event: MouseEvent<HTMLButtonElement>) => {
    event.currentTarget.style.setProperty("--mag-x", "0px");
    event.currentTarget.style.setProperty("--mag-y", "0px");
  };

  return (
    <button className={`magnetic ${className}`} onClick={onClick} onMouseMove={handleMove} onMouseLeave={reset}>
      {children}
    </button>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("approach");
  const heroVisualRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);

  const scrollTo = (id: string) => {
    setMenuOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    let frame = 0;

    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const doc = document.documentElement;
        const maxScroll = Math.max(1, doc.scrollHeight - window.innerHeight);
        const progress = Math.min(1, Math.max(0, window.scrollY / maxScroll));

        doc.style.setProperty("--scroll-progress", progress.toFixed(4));
        doc.style.setProperty("--scroll-px", `${progress * 100}%`);
        doc.style.setProperty("--bg-shift", `${progress * -3}%`);
        doc.style.setProperty("--ambient-one-y", `${progress * 210}px`);
        doc.style.setProperty("--ambient-two-y", `${progress * -170}px`);
        doc.style.setProperty("--green-light-y", `${18 + progress * 58}%`);
        doc.style.setProperty("--forest-light-y", `${76 - progress * 38}%`);

        const y = window.scrollY;
        if (heroVisualRef.current) {
          heroVisualRef.current.style.setProperty("--hero-scroll", `${Math.min(y * 0.17, 180)}px`);
        }
      });
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  useEffect(() => {
    const onMove = (event: globalThis.MouseEvent) => {
      document.documentElement.style.setProperty("--pointer-x", `${event.clientX}px`);
      document.documentElement.style.setProperty("--pointer-y", `${event.clientY}px`);
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
        cursorRef.current.classList.add("cursor-visible");
      }

      if (heroVisualRef.current && window.innerWidth > 760) {
        const rect = heroVisualRef.current.getBoundingClientRect();
        const dx = (event.clientX - (rect.left + rect.width / 2)) / rect.width;
        const dy = (event.clientY - (rect.top + rect.height / 2)) / rect.height;
        heroVisualRef.current.style.setProperty("--tilt-x", `${dy * -7}deg`);
        heroVisualRef.current.style.setProperty("--tilt-y", `${dx * 8}deg`);
      }
    };

    const onLeave = () => cursorRef.current?.classList.remove("cursor-visible");
    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    return () => {
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  useEffect(() => {
    const nodes = sections
      .map((section) => document.getElementById(section.id))
      .filter((node): node is HTMLElement => Boolean(node));

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActiveSection(entry.target.id);
        });
      },
      { rootMargin: "-38% 0px -48% 0px", threshold: 0 },
    );

    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  const tiltCard = (event: MouseEvent<HTMLDivElement>) => {
    if (window.innerWidth < 760 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    event.currentTarget.style.setProperty("--tilt-y", `${(px - 0.5) * 7}deg`);
    event.currentTarget.style.setProperty("--tilt-x", `${(0.5 - py) * 7}deg`);
    event.currentTarget.style.setProperty("--spot-x", `${px * 100}%`);
    event.currentTarget.style.setProperty("--spot-y", `${py * 100}%`);
  };

  const resetCard = (event: MouseEvent<HTMLDivElement>) => {
    event.currentTarget.style.setProperty("--tilt-x", "0deg");
    event.currentTarget.style.setProperty("--tilt-y", "0deg");
  };

  return (
    <main className="site-shell">
      <div className="scroll-progress" aria-hidden="true" />
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <div ref={cursorRef} className="cursor-aura" aria-hidden="true" />

      <header className="site-nav">
        <a className="wordmark" href="#" aria-label="RetinaScope home">retina<span>scope</span><sup>®</sup></a>

        <nav className="desktop-nav" aria-label="Main navigation">
          {sections.map((section) => (
            <button
              key={section.id}
              className={`nav-link ${activeSection === section.id ? "nav-link-active" : ""}`}
              onClick={() => scrollTo(section.id)}
            >
              {section.eyebrow.replace("THE ", "")}
            </button>
          ))}
        </nav>

        <MagneticButton className="nav-cta" onClick={() => scrollTo("platform")}>
          ENTER RETINASCOPE <span className="arrow">↗</span>
        </MagneticButton>

        <button
          className="mobile-menu"
          onClick={() => setMenuOpen((open) => !open)}
          aria-expanded={menuOpen}
          aria-label="Toggle navigation"
        >
          <span>{menuOpen ? "CLOSE" : "MENU"}</span>
          <i className={menuOpen ? "menu-icon menu-open" : "menu-icon"} />
        </button>
      </header>

      {menuOpen && (
        <div className="mobile-nav">
          {sections.map((section) => (
            <button key={section.id} onClick={() => scrollTo(section.id)}>{section.eyebrow}</button>
          ))}
          <button onClick={() => scrollTo("platform")}>ENTER RETINASCOPE ↗</button>
        </div>
      )}

      <section className="hero">
        <div className="hero-graduated-line" aria-hidden="true" />
        <div className="hero-content">
          <Reveal className="hero-kicker">
            <span className="kicker-line" />
            EXPLAINABLE AI <b>·</b> RETINAL INTELLIGENCE
          </Reveal>

          <h1 className="hero-title">
            <span className="title-line title-line-one">A clearer</span>
            <span className="title-line title-line-two">vision of</span>
            <em className="title-line title-line-three">what&apos;s ahead.</em>
          </h1>

          <div className="hero-bottom">
            <p className="hero-description">
              Bringing explainable artificial intelligence to diabetic retinopathy screening.
              Designed around clarity, precision and the importance of early insight.
            </p>
            <button className="hero-link" onClick={() => scrollTo("approach")}>
              EXPLORE THE PLATFORM <span>↗</span>
            </button>
          </div>
        </div>

        <div className="hero-visual" ref={heroVisualRef} aria-hidden="true">
          <div className="hero-visual-inner">
            <div className="orbit orbit-outer"><span className="orbit-marker marker-one" /><span className="orbit-marker marker-two" /></div>
            <div className="orbit orbit-middle" />
            <div className="orbit orbit-inner" />
            <div className="orbit axis axis-horizontal" />
            <div className="orbit axis axis-vertical" />
            <div className="retina-core"><div className="retina-core-inner" /><span className="core-spark" /></div>
            <div className="orbit-caption caption-top">RETINAL FIELD / 01</div>
            <div className="orbit-caption caption-bottom">SIGNAL ANALYSIS <span>●</span></div>
            <div className="orbit-scan" />
          </div>
        </div>

        <div className="hero-meta">
          <span>RETINASCOPE® / RESEARCH INTERFACE</span>
          <button onClick={() => scrollTo("approach")}>SCROLL TO EXPLORE <span>↓</span></button>
        </div>
      </section>

      <section className="manifesto">
        <Reveal className="manifesto-heading">
          <p className="section-kicker">A DIFFERENT WAY TO SEE THE DATA</p>
          <h2>Technology should not<br />obscure the <em>picture.</em></h2>
        </Reveal>
        <Reveal className="manifesto-note" delay={130}>
          <span className="small-rule" />
          <p>RetinaScope pairs model-driven screening with visual explanations, so every result has context — not just a number.</p>
          <span className="note-index">A MORE HUMAN VIEW OF MACHINE INTELLIGENCE</span>
        </Reveal>
      </section>

      {sections.map((section, index) => (
        <section className={`story-section story-${index + 1}`} id={section.id} key={section.id}>
          <div className="story-topline">
            <span>{section.number} / 03</span>
            <span>{section.eyebrow}</span>
          </div>
          <div className="story-layout">
            <Reveal className="story-copy">
              <p className="section-kicker"><span className="kicker-dot" />{section.tag}</p>
              <h2>{section.title.map((line, lineIndex) => <span key={lineIndex}>{line}{lineIndex === 0 ? <br /> : null}</span>)}</h2>
              <p className="story-description">{section.body}</p>
              <button className="story-link" onClick={() => scrollTo("platform")}>DISCOVER THE PLATFORM <span>↗</span></button>
            </Reveal>

            <Reveal className="visual-reveal" delay={120}>
              <div className={`story-visual story-visual-${index + 1}`} onMouseMove={tiltCard} onMouseLeave={resetCard}>
                {index === 0 && (
                  <>
                    <div className="visual-coordinate coord-one">FIELD / 01</div>
                    <div className="target-rings"><i /><i /><i /><b /></div>
                    <div className="target-cross target-cross-x" /><div className="target-cross target-cross-y" />
                    <div className="target-sweep" />
                    <div className="visual-bottom-label"><span>CONTEXT</span><span>CLARITY</span><span>INSIGHT</span></div>
                  </>
                )}
                {index === 1 && (
                  <>
                    <div className="retinal-grid" />
                    <div className="retinal-image-art"><div className="retinal-vessel vessel-a" /><div className="retinal-vessel vessel-b" /><div className="retinal-vessel vessel-c" /><div className="retinal-vessel vessel-d" /><div className="retinal-disc" /></div>
                    <div className="retinal-focus" />
                    <div className="visual-coordinate coord-two">GRAD-CAM / ATTENTION</div>
                    <div className="visual-bottom-label"><span>MODEL OUTPUT</span><span className="live-status">● EXPLAINABILITY</span></div>
                  </>
                )}
                {index === 2 && (
                  <>
                    <div className="impact-top"><span>CLASSIFICATION</span><span>FIVE LEVELS</span></div>
                    <div className="impact-number">05<span> / grades</span></div>
                    <div className="impact-chart">
                      {[34, 52, 77, 61, 94].map((height, barIndex) => <i key={barIndex} style={{ "--bar-height": `${height}%`, animationDelay: `${barIndex * 90}ms` } as CSSProperties}><b>{`0${barIndex}`}</b></i>)}
                    </div>
                    <div className="visual-bottom-label"><span>IMAGE TO INSIGHT</span><span>RESEARCH PROTOTYPE</span></div>
                  </>
                )}
              </div>
            </Reveal>
          </div>
        </section>
      ))}

      <section className="platform-section" id="platform">
        <Reveal>
          <p className="section-kicker"><span className="kicker-dot" /> THE RETINASCOPE PLATFORM</p>
          <h2>See the details.<br /><em>Move forward.</em></h2>
          <p className="platform-description">A focused workspace for retinal image screening, explainable results and planning insights.</p>
          <MagneticButton className="platform-button" onClick={() => window.open("http://127.0.0.1:8000/docs", "_blank", "noopener,noreferrer")}>
            OPEN API DOCUMENTATION <span className="arrow">↗</span>
          </MagneticButton>
        </Reveal>
        <footer className="site-footer">
          <a className="wordmark footer-wordmark" href="#">retina<span>scope</span><sup>®</sup></a>
          <span>EXPLAINABLE AI FOR RETINAL SCREENING</span>
          <span>© 2026 RETINASCOPE</span>
        </footer>
        <p className="disclaimer">RetinaScope is a research and demonstration prototype. AI-assisted screening outputs are not standalone clinical diagnoses and should not replace evaluation by a qualified eye-care professional.</p>
      </section>
    </main>
  );
}
