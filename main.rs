mod chat;
mod opts;

use opts::{Cmd, Opts};

fn main() {
    env_logger::init();
    let opts: Opts = argh::from_env();

    match opts.command {
        Cmd::Train(cmd) => {
            noodle::train(&cmd.input, &cmd.output, cmd.backend.into(), cmd.max_epochs)
                .expect("training failed");
        }
        Cmd::Finetune(cmd) => {
            noodle::finetune(
                &cmd.model,
                &cmd.input,
                &cmd.output,
                cmd.backend.into(),
                cmd.max_epochs,
            )
            .expect("fine-tuning failed");
        }
        Cmd::Eval(cmd) => {
            noodle::eval(&cmd.model, &cmd.corpus, cmd.backend.into()).expect("evaluation failed");
        }
        Cmd::Chat(cmd) => {
            chat::chat(&cmd.model, cmd.backend.into()).expect("chat failed");
        }
    }
}
