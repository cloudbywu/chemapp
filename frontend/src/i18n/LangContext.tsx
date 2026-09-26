import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";
import { translations, type Lang, type TranslationSchema } from "./translations";

type LangContextType = {
  lang: Lang;
  t: TranslationSchema;
  toggleLang: () => void;
};

const LangContext = createContext<LangContextType | null>(null);

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    const saved = localStorage.getItem("chemapp-lang");
    return (saved === "zh" || saved === "en") ? saved : "zh";
  });

  const toggleLang = useCallback(() => {
    setLang((prev) => {
      const next = prev === "zh" ? "en" : "zh";
      localStorage.setItem("chemapp-lang", next);
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = `${translations[lang].app.title} — ${translations[lang].app.subtitle}`;
  }, [lang]);

  const value: LangContextType = {
    lang,
    t: translations[lang],
    toggleLang,
  };

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang must be used within LangProvider");
  return ctx;
}
