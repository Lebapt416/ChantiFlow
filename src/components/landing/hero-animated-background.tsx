'use client';

import { motion } from 'framer-motion';

export function AnimatedBackground() {
  return (
    <>
      <motion.div
        className="absolute left-1/4 top-1/4 w-[600px] h-[600px] bg-emerald-500/20 dark:bg-emerald-400/20 rounded-full blur-3xl"
        animate={{
          x: [0, 100, 0],
          y: [0, 50, 0],
          scale: [1, 1.2, 1],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute right-1/4 top-1/3 w-[500px] h-[500px] bg-blue-500/20 dark:bg-blue-400/20 rounded-full blur-3xl"
        animate={{
          x: [0, -80, 0],
          y: [0, 80, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute left-1/2 bottom-1/4 w-[550px] h-[550px] bg-purple-500/20 dark:bg-purple-400/20 rounded-full blur-3xl"
        animate={{
          x: [0, 60, 0],
          y: [0, -60, 0],
          scale: [1, 1.15, 1],
        }}
        transition={{
          duration: 22,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
    </>
  );
}

